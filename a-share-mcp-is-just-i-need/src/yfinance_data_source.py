# 使用 yfinance (Yahoo Finance) 实现 FinancialDataSource 接口的美股数据源
#
# 设计说明:
# - 继承 BaostockDataSource, 直接复用其中的情感/风险本地模型方法
#   (_load_risk_model / _load_sentiment_model / _analyze_risk / _analyze_sentiment)。
#   这些模型是用 NASDAQ 新闻训练的, 天然适配美股。
# - 覆盖所有"数据获取"方法, 改用 yfinance 拉取美股数据, 并返回与 baostock
#   相同列名的 DataFrame, 从而无需改动上层 analysis.py 等工具的报告逻辑。
# - A 股特有、美股无对应的方法(中国宏观利率、货币供应、上证50/沪深300等)
#   直接返回空 DataFrame, 避免误触发 baostock 联网(会长时间超时)。

import pandas as pd
import yfinance as yf
from typing import List, Optional
import logging
from datetime import datetime

from .baostock_data_source import BaostockDataSource
from .data_source_interface import NoDataFoundError, DataSourceError

logger = logging.getLogger(__name__)

# yfinance 频率映射
_FREQ_MAP = {"d": "1d", "w": "1wk", "m": "1mo",
             "1d": "1d", "1wk": "1wk", "1mo": "1mo"}


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _clean_code(code: str) -> str:
    """把 'sh.600000' / 'aapl' 之类归一化为 yfinance 用的 ticker。
    美股一般是纯字母 ticker(AAPL/TSLA)。若误带了 A 股前缀则去掉。"""
    if not code:
        return code
    c = code.strip().upper()
    for pre in ("SH.", "SZ.", "SH", "SZ"):
        if c.startswith(pre) and c[len(pre):].isdigit():
            c = c[len(pre):]
            break
    return c


class YFinanceDataSource(BaostockDataSource):
    """基于 Yahoo Finance 的美股数据源。复用父类的本地情感/风险模型。"""

    # ---------------- 行情 K 线 ----------------
    def get_historical_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        ticker = _clean_code(code)
        interval = _FREQ_MAP.get(frequency, "1d")
        logger.info(
            f"[yfinance] history {ticker} {start_date}~{end_date} interval={interval}")
        try:
            t = yf.Ticker(ticker)
            # yfinance 的 end 是开区间, 加一天保证包含 end_date
            end_inc = (pd.to_datetime(end_date) + pd.Timedelta(days=1)
                       ).strftime("%Y-%m-%d")
            hist = t.history(start=start_date, end=end_inc,
                             interval=interval, auto_adjust=True)
        except Exception as e:
            raise DataSourceError(f"yfinance 获取 {ticker} K线失败: {e}")

        if hist is None or hist.empty:
            raise NoDataFoundError(f"未找到 {ticker} 在该区间的行情数据")

        hist = hist.reset_index()
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(hist[date_col]).dt.strftime("%Y-%m-%d")
        out["code"] = ticker
        out["open"] = hist["Open"].round(4)
        out["high"] = hist["High"].round(4)
        out["low"] = hist["Low"].round(4)
        out["close"] = hist["Close"].round(4)
        out["preclose"] = hist["Close"].shift(1).round(4)
        out["volume"] = hist["Volume"]
        out["amount"] = (hist["Close"] * hist["Volume"]).round(2)
        out["pctChg"] = (hist["Close"].pct_change() * 100).round(4)
        # 保持按日期升序 (analysis.py 依赖 iloc[0]=最早, iloc[-1]=最新)
        return out

    # ---------------- 基本信息 ----------------
    def get_stock_basic_info(self, code: str, fields: Optional[List[str]] = None) -> pd.DataFrame:
        ticker = _clean_code(code)
        logger.info(f"[yfinance] basic_info {ticker}")
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as e:
            raise DataSourceError(f"yfinance 获取 {ticker} 基本信息失败: {e}")

        name = info.get("longName") or info.get("shortName")
        if not name:
            raise NoDataFoundError(f"未找到 ticker {ticker} 的基本信息")

        ipo = ""
        epoch = info.get("firstTradeDateEpochUtc")
        if epoch:
            try:
                ipo = datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%d")
            except Exception:
                ipo = ""

        row = {
            "code": ticker,
            "code_name": name,
            "industry": info.get("industry") or info.get("sector") or "未知",
            "sector": info.get("sector") or "",
            "ipoDate": ipo,
            "tradeStatus": "1",
            # 美股估值/规模, 供 value_agent 使用 (额外列不影响既有逻辑)
            "marketCap": info.get("marketCap"),
            "peTTM": info.get("trailingPE"),
            "pbMRQ": info.get("priceToBook"),
            "currentPrice": info.get("currentPrice"),
        }
        return pd.DataFrame([row])

    # ---------------- 财务: 盈利能力 ----------------
    def get_profit_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        ticker = _clean_code(code)
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as e:
            raise DataSourceError(f"yfinance 获取 {ticker} 盈利数据失败: {e}")
        data = {}
        if info.get("returnOnEquity") is not None:
            data["roeAvg"] = round(info["returnOnEquity"] * 100, 2)
        if info.get("profitMargins") is not None:
            data["npMargin"] = round(info["profitMargins"] * 100, 2)
        if info.get("grossMargins") is not None:
            data["gpMargin"] = round(info["grossMargins"] * 100, 2)
        if info.get("netIncomeToCommon") is not None:
            data["netProfit"] = info["netIncomeToCommon"]
        if not data:
            return _empty_df()
        return pd.DataFrame([data])

    # ---------------- 财务: 成长能力 ----------------
    def get_growth_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        ticker = _clean_code(code)
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
        except Exception as e:
            raise DataSourceError(f"yfinance 获取 {ticker} 成长数据失败: {e}")
        data = {}
        if info.get("earningsGrowth") is not None:
            data["YOYNI"] = round(info["earningsGrowth"] * 100, 2)  # 净利润同比
        if info.get("revenueGrowth") is not None:
            data["YOYRevenue"] = round(info["revenueGrowth"] * 100, 2)
        # 尝试从资产负债表算净资产/总资产同比
        try:
            bs = t.balance_sheet
            if bs is not None and bs.shape[1] >= 2:
                def yoy(label):
                    if label in bs.index:
                        cur, prev = bs.loc[label].iloc[0], bs.loc[label].iloc[1]
                        if prev and prev != 0:
                            return round((cur / prev - 1) * 100, 2)
                    return None
                eq = yoy("Stockholders Equity") or yoy("Total Equity Gross Minority Interest")
                at = yoy("Total Assets")
                if eq is not None:
                    data["YOYEquity"] = eq
                if at is not None:
                    data["YOYAsset"] = at
        except Exception as e:
            logger.warning(f"计算 {ticker} 资产成长率失败: {e}")
        if not data:
            return _empty_df()
        return pd.DataFrame([data])

    # ---------------- 财务: 偿债能力 ----------------
    def get_balance_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        ticker = _clean_code(code)
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
        except Exception as e:
            raise DataSourceError(f"yfinance 获取 {ticker} 偿债数据失败: {e}")
        data = {}
        if info.get("currentRatio") is not None:
            data["currentRatio"] = info["currentRatio"]
        if info.get("quickRatio") is not None:
            data["quickRatio"] = info["quickRatio"]
        # 资产负债率 = 总负债 / 总资产 * 100
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                tl = bs.loc["Total Liabilities Net Minority Interest"].iloc[0] \
                    if "Total Liabilities Net Minority Interest" in bs.index else None
                ta = bs.loc["Total Assets"].iloc[0] \
                    if "Total Assets" in bs.index else None
                if tl is not None and ta:
                    data["assetLiabRatio"] = round(tl / ta * 100, 2)
        except Exception as e:
            logger.warning(f"计算 {ticker} 资产负债率失败: {e}")
        if "assetLiabRatio" not in data and info.get("debtToEquity") is not None:
            # 退而求其次: 用 debtToEquity 粗略换算负债率
            de = info["debtToEquity"] / 100.0
            data["assetLiabRatio"] = round(de / (1 + de) * 100, 2)
        if not data:
            return _empty_df()
        return pd.DataFrame([data])

    # ---------------- 财务: 杜邦(报告未直接引用其列, 返回空即可) ----------------
    def get_dupont_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return _empty_df()

    # ---------------- 财务: 现金流 ----------------
    def get_cash_flow_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        ticker = _clean_code(code)
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as e:
            raise DataSourceError(f"yfinance 获取 {ticker} 现金流失败: {e}")
        data = {}
        if info.get("operatingCashflow") is not None:
            data["operatingCashFlow"] = info["operatingCashflow"]
        if info.get("freeCashflow") is not None:
            data["freeCashFlow"] = info["freeCashflow"]
        if not data:
            return _empty_df()
        return pd.DataFrame([data])

    def get_operation_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return _empty_df()

    # ---------------- 行业 ----------------
    def get_stock_industry(self, code: Optional[str] = None, date: Optional[str] = None) -> pd.DataFrame:
        # 无 code 时(用于全行业对比)无法枚举全部美股, 返回空
        if not code:
            return _empty_df()
        ticker = _clean_code(code)
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            return _empty_df()
        return pd.DataFrame([{
            "code": ticker,
            "code_name": info.get("longName") or info.get("shortName") or ticker,
            "industry": info.get("industry") or info.get("sector") or "未知",
        }])

    # ---------------- 交易日历 ----------------
    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        try:
            end = end_date or datetime.now().strftime("%Y-%m-%d")
            start = start_date or (pd.to_datetime(end) -
                                   pd.Timedelta(days=15)).strftime("%Y-%m-%d")
            end_inc = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            hist = yf.Ticker("SPY").history(start=start, end=end_inc, interval="1d")
            days = pd.to_datetime(hist.index).strftime("%Y-%m-%d").tolist()
            return pd.DataFrame({
                "calendar_date": days,
                "is_trading_day": ["1"] * len(days),
            })
        except Exception as e:
            logger.warning(f"获取美股交易日历失败: {e}")
            return _empty_df()

    # ---------------- 分红 ----------------
    def get_dividend_data(self, code: str, year: str, year_type: str = "report") -> pd.DataFrame:
        ticker = _clean_code(code)
        try:
            div = yf.Ticker(ticker).dividends
            if div is None or div.empty:
                return _empty_df()
            df = div.reset_index()
            df.columns = ["date", "dividend"]
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            if year:
                df = df[df["date"].str.startswith(str(year))]
            return df
        except Exception as e:
            logger.warning(f"获取 {ticker} 分红失败: {e}")
            return _empty_df()

    # ---------------- 新闻 + 情感/风险分析 ----------------
    def crawl_news(self, query: str, top_k: int = 3) -> str:
        """用 Yahoo Finance 抓取美股新闻。情感/风险模型已停用时自动跳过分析。"""
        try:
            # 新闻数量上限固定为 3 条, 避免过多新闻拖慢流程
            top_k = max(1, min(top_k, 3))
            risk_model, risk_tokenizer = self._load_risk_model()
            sentiment_model, sentiment_tokenizer = self._load_sentiment_model()

            ticker = _clean_code(query)
            items = []
            try:
                items = yf.Ticker(ticker).news or []
            except Exception as e:
                logger.warning(f"yfinance news({ticker}) 失败: {e}")
            # 若按 ticker 拿不到, 尝试搜索
            if not items:
                try:
                    items = yf.Search(query, news_count=top_k).news or []
                except Exception as e:
                    logger.warning(f"yfinance Search({query}) 失败: {e}")

            if not items:
                return f"未找到与 '{query}' 相关的新闻。"

            lines = [f"找到以下与 '{query}' 相关的新闻：\n"]
            for i, it in enumerate(items[:top_k], 1):
                c = it.get("content", it) if isinstance(it, dict) else {}
                title = c.get("title") or ""
                summary = c.get("summary") or c.get("description") or ""
                provider = ""
                if isinstance(c.get("provider"), dict):
                    provider = c["provider"].get("displayName", "")
                link = ""
                if isinstance(c.get("canonicalUrl"), dict):
                    link = c["canonicalUrl"].get("url", "")
                if not title:
                    continue

                text = f"{title}. {summary}".strip()
                risk = self._analyze_risk(text, risk_model, risk_tokenizer) \
                    if risk_model else "未分析"
                sentiment = self._analyze_sentiment(text, sentiment_model, sentiment_tokenizer) \
                    if sentiment_model else "未分析"

                lines.append(f"{i}. {title}")
                lines.append(f"   来源: {provider or 'Yahoo Finance'}")
                if summary:
                    lines.append(f"   摘要: {summary[:200]}")
                lines.append(f"   情感分析: {sentiment}")
                lines.append(f"   风险分析: {risk}")
                if link:
                    lines.append(f"   链接: {link}")
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            logger.exception(f"crawl_news 失败: {e}")
            return f"爬取新闻时出错: {e}"

    # ---------------- A股特有、美股无对应 -> 返回空, 避免误触发 baostock 联网 ----------------
    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        return _empty_df()

    def get_deposit_rate_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        return _empty_df()

    def get_loan_rate_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        return _empty_df()

    def get_required_reserve_ratio_data(self, start_date=None, end_date=None, year_type: str = '0') -> pd.DataFrame:
        return _empty_df()

    def get_money_supply_data_month(self, start_date=None, end_date=None) -> pd.DataFrame:
        return _empty_df()

    def get_money_supply_data_year(self, start_date=None, end_date=None) -> pd.DataFrame:
        return _empty_df()

    def get_sz50_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return _empty_df()

    def get_hs300_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return _empty_df()

    def get_zz500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return _empty_df()

    def get_adjust_factor_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return _empty_df()

    def get_performance_express_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return _empty_df()

    def get_forecast_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return _empty_df()
