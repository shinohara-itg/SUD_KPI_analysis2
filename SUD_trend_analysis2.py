import os
import uuid
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import textwrap

# =========================
# Azure OpenAI 設定
# =========================
load_dotenv()


def get_azure_client() -> Optional[AzureOpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not api_key or not azure_endpoint or not api_version:
        return None

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )


CLIENT = get_azure_client()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# =========================
# 画面設定
# =========================
st.set_page_config(
    page_title="施設KPI分析ツール",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 定数
# =========================
REQUIRED_COLUMNS = [
    "調査年度",
    "対象施設名",
    "指標No",
    "指標名",
    "セグメントNo",
    "セグメント名",
    "指標",
]

TOTAL_LABEL = "TOTAL"


# =========================
# Session State
# =========================
def init_session_state() -> None:
    defaults = {
        "mode": None,
        "raw_df": None,
        "uploaded_file_name": None,
        "selected_metric_for_segment": None,
        "selected_metric_for_facility": None,
        "selected_segment_for_comment": None,
        "selected_facility_for_comment": None,
        "selected_segment_for_trend": None,
        "block1_comment": None,
        "block2_generated_comment": None,
        "block3_generated_comment": None,
        "segment_mode_generated_comment": None,
        "overview_summary_comment": None,
        "overview_navigation_comment": None,
        "overview_block2_summary_comment": None,
        "overview_block2_navigation_comment": None,
        "overview_block3_summary_comment": None,
        "overview_block3_navigation_comment": None,
        "current_comment_card": None,
        "draft_selected_year": None,
        "draft_selected_facility": None,
        "active_selected_year": None,
        "active_selected_facility": None,
        "pinned_comments": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



def reset_analysis_states() -> None:
    st.session_state.raw_df = None
    st.session_state.selected_metric_for_segment = None
    st.session_state.selected_metric_for_facility = None
    st.session_state.selected_segment_for_comment = None
    st.session_state.selected_facility_for_comment = None
    st.session_state.selected_segment_for_trend = None
    st.session_state.block1_comment = None
    st.session_state.block2_generated_comment = None
    st.session_state.block3_generated_comment = None
    st.session_state.segment_mode_generated_comment = None
    st.session_state.overview_summary_comment = None
    st.session_state.overview_navigation_comment = None
    st.session_state.overview_block2_summary_comment = None
    st.session_state.overview_block2_navigation_comment = None
    st.session_state.overview_block3_summary_comment = None
    st.session_state.overview_block3_navigation_comment = None
    st.session_state.current_comment_card = None
    st.session_state.draft_selected_year = None
    st.session_state.draft_selected_facility = None
    st.session_state.active_selected_year = None
    st.session_state.active_selected_facility = None
    st.session_state.pinned_comments = []


init_session_state()


# =========================
# 共通関数
# =========================
def set_mode(mode_name: str) -> None:
    st.session_state.mode = mode_name


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("対応していないファイル形式です。")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"必要な列が不足しています: {', '.join(missing)}")

    work_df = df.copy()
    work_df["調査年度"] = work_df["調査年度"].astype(str).str.strip()
    work_df["対象施設名"] = work_df["対象施設名"].astype(str).str.strip()
    work_df["指標名"] = work_df["指標名"].astype(str).str.strip()
    work_df["セグメント名"] = work_df["セグメント名"].astype(str).str.strip()
    work_df["指標No"] = pd.to_numeric(work_df["指標No"], errors="coerce")
    work_df["セグメントNo"] = pd.to_numeric(work_df["セグメントNo"], errors="coerce")
    work_df["指標"] = pd.to_numeric(work_df["指標"], errors="coerce")

    return work_df



def get_year_options(df: pd.DataFrame) -> list[str]:
    years = df["調査年度"].dropna().astype(str).drop_duplicates().tolist()
    return sorted(years)



def get_previous_year(selected_year: str, year_options_asc: list[str]) -> Optional[str]:
    if selected_year not in year_options_asc:
        return None

    idx = year_options_asc.index(selected_year)
    if idx == 0:
        return None
    return year_options_asc[idx - 1]



def get_facility_options(df: pd.DataFrame) -> list[str]:
    facilities = df["対象施設名"].dropna().astype(str).drop_duplicates().tolist()
    return sorted(facilities)



def is_total_segment(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().eq(TOTAL_LABEL)



def format_df_for_prompt(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "該当データなし"
    return df.head(max_rows).to_csv(index=False)

def render_readable_table(
    df: pd.DataFrame,
    display_columns: list[str],
    decimal_columns: Optional[list[str]] = None,
    font_size_px: int = 16,
) -> None:
    if df is None or df.empty:
        st.info("該当データがありません。")
        return

    table_df = df[display_columns].copy()

    if decimal_columns is None:
        decimal_columns = []

    format_dict = {}
    for col in decimal_columns:
        if col in table_df.columns:
            format_dict[col] = "{:.1f}"

    styled_df = (
        table_df.style
        .format(format_dict, na_rep="-")
        .set_properties(**{
            "font-size": f"{font_size_px}px",
            "white-space": "normal",
        })
    )

    st.table(styled_df)


def format_diff_value_text(value) -> str:
    if pd.isna(value):
        return "-"

    return f"{float(value):.1f}"


def format_diff_value_text(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1f}"


def build_segment_metrics_html_table(metric_cols: list[str], seg_row: pd.Series) -> str:
    header_html = "".join([
        (
            f'<th style="min-width:72px; padding:6px 8px; text-align:center; '
            f'font-size:16px; font-weight:700; border-bottom:1px solid #ddd; '
            f'white-space:nowrap;">{col}</th>'
        )
        for col in metric_cols
    ])

    value_cells = []
    for col in metric_cols:
        value = seg_row[col]
        color = "#d62728" if pd.notna(value) and float(value) < 0 else "#111111"
        text = format_diff_value_text(value)
        value_cells.append(
            f'<td style="min-width:72px; padding:8px 8px; text-align:center; '
            f'font-size:22px; font-weight:700; color:{color}; '
            f'white-space:nowrap;">{text}</td>'
        )

    values_html = "".join(value_cells)

    html = f"""
<div style="overflow-x:auto; width:100%;">
  <table style="border-collapse:collapse; width:max-content; min-width:100%;">
    <thead>
      <tr>{header_html}</tr>
    </thead>
    <tbody>
      <tr>{values_html}</tr>
    </tbody>
  </table>
</div>
"""
    return textwrap.dedent(html).strip()

# =========================
# KPI変化モード用集計関数
# =========================
def format_diff_value_html(value) -> str:
    if pd.isna(value):
        return '<span style="font-size:22px; font-weight:700;">-</span>'

    color = "#d62728" if float(value) < 0 else "#111111"
    return (
        f'<span style="font-size:22px; font-weight:700; color:{color};">'
        f'{float(value):.1f}'
        f"</span>"
    )

def build_block1_metric_diff_table(
    df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
) -> pd.DataFrame:
    mask = (
        df["調査年度"].isin([selected_year, previous_year])
        & (df["対象施設名"] == selected_facility)
        & is_total_segment(df["セグメント名"])
    )
    work_df = df.loc[mask].copy()

    if work_df.empty:
        return pd.DataFrame()

    pivot_df = pd.pivot_table(
        work_df,
        index=["指標No", "指標名"],
        columns="調査年度",
        values="指標",
        aggfunc="first",
    ).reset_index()

    if selected_year not in pivot_df.columns or previous_year not in pivot_df.columns:
        return pd.DataFrame()

    diff_col = f"前年差（{selected_year}-{previous_year}）"
    pivot_df[diff_col] = (pivot_df[selected_year] - pivot_df[previous_year]).round(1)

    result_df = pivot_df[["指標No", "指標名", previous_year, selected_year, diff_col]].copy()
    result_df = result_df.sort_values(["指標No", "指標名"]).reset_index(drop=True)
    return result_df



def build_block2_segment_diff_table(
    df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
    metric_no: int,
) -> pd.DataFrame:
    mask = (
        df["調査年度"].isin([selected_year, previous_year])
        & (df["対象施設名"] == selected_facility)
        & (df["指標No"] == metric_no)
        & (~is_total_segment(df["セグメント名"]))
    )
    work_df = df.loc[mask].copy()

    if work_df.empty:
        return pd.DataFrame()

    pivot_df = pd.pivot_table(
        work_df,
        index=["セグメントNo", "セグメント名"],
        columns="調査年度",
        values="指標",
        aggfunc="first",
    ).reset_index()

    if selected_year not in pivot_df.columns or previous_year not in pivot_df.columns:
        return pd.DataFrame()

    diff_col = f"前年差（{selected_year}-{previous_year}）"
    pivot_df[diff_col] = (pivot_df[selected_year] - pivot_df[previous_year]).round(1)

    result_df = pivot_df[["セグメントNo", "セグメント名", previous_year, selected_year, diff_col]].copy()
    result_df = result_df.sort_values(["セグメントNo", "セグメント名"]).reset_index(drop=True)
    return result_df



def build_block3_facility_diff_table(
    df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    metric_no: int,
) -> pd.DataFrame:
    mask = (
        df["調査年度"].isin([selected_year, previous_year])
        & (df["指標No"] == metric_no)
        & is_total_segment(df["セグメント名"])
    )
    work_df = df.loc[mask].copy()

    if work_df.empty:
        return pd.DataFrame()

    pivot_df = pd.pivot_table(
        work_df,
        index=["対象施設名"],
        columns="調査年度",
        values="指標",
        aggfunc="first",
    ).reset_index()

    if selected_year not in pivot_df.columns or previous_year not in pivot_df.columns:
        return pd.DataFrame()

    diff_col = f"前年差（{selected_year}-{previous_year}）"
    pivot_df[diff_col] = (pivot_df[selected_year] - pivot_df[previous_year]).round(1)

    result_df = pivot_df[["対象施設名", previous_year, selected_year, diff_col]].copy()
    result_df = result_df.sort_values([diff_col, "対象施設名"], ascending=[False, True]).reset_index(drop=True)
    return result_df


# =========================
# セグメント分析モード用集計関数
# =========================
def build_segment_analysis_diff_table(
    df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
) -> pd.DataFrame:
    mask = (
        df["調査年度"].isin([selected_year, previous_year])
        & (df["対象施設名"] == selected_facility)
        & (~is_total_segment(df["セグメント名"]))
    )
    work_df = df.loc[mask].copy()

    if work_df.empty:
        return pd.DataFrame()

    metric_master = (
        work_df[["指標No", "指標名"]]
        .drop_duplicates()
        .sort_values(["指標No", "指標名"])
        .reset_index(drop=True)
    )

    base_df = (
        work_df[["セグメントNo", "セグメント名"]]
        .drop_duplicates()
        .sort_values(["セグメントNo", "セグメント名"])
        .reset_index(drop=True)
    )

    result_df = base_df.copy()

    for _, metric_row in metric_master.iterrows():
        metric_no = metric_row["指標No"]
        metric_name = metric_row["指標名"]

        metric_df = work_df[work_df["指標No"] == metric_no].copy()
        if metric_df.empty:
            continue

        pivot_df = pd.pivot_table(
            metric_df,
            index=["セグメントNo", "セグメント名"],
            columns="調査年度",
            values="指標",
            aggfunc="first",
        ).reset_index()

        if selected_year not in pivot_df.columns or previous_year not in pivot_df.columns:
            continue

        diff_col_name = f"{metric_name}"
        pivot_df[diff_col_name] = (pivot_df[selected_year] - pivot_df[previous_year]).round(1)
        diff_df = pivot_df[["セグメントNo", "セグメント名", diff_col_name]].copy()
        result_df = result_df.merge(

            diff_df,
            on=["セグメントNo", "セグメント名"],
            how="left",
        )

    metric_diff_cols = [c for c in result_df.columns if c not in ["セグメントNo", "セグメント名"]]
    if not metric_diff_cols:
        return pd.DataFrame()

    result_df = result_df.sort_values(["セグメントNo", "セグメント名"]).reset_index(drop=True)
    return result_df

# =========================
# 総括モード用集計関数
# =========================
def get_diff_col_name(selected_year: str, previous_year: str) -> str:
    return f"前年差（{selected_year}-{previous_year}）"


def build_overview_top_metric_tables(
    metric_diff_df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metric_diff_df is None or metric_diff_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    diff_col = get_diff_col_name(selected_year, previous_year)
    work_df = metric_diff_df.copy()
    positive_df = work_df.sort_values([diff_col, "指標No", "指標名"], ascending=[False, True, True]).head(top_n).reset_index(drop=True)
    negative_df = work_df.sort_values([diff_col, "指標No", "指標名"], ascending=[True, True, True]).head(top_n).reset_index(drop=True)
    return positive_df, negative_df


def build_overview_segment_summary_tables(
    df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = (
        df["調査年度"].isin([selected_year, previous_year])
        & (df["対象施設名"] == selected_facility)
        & (~is_total_segment(df["セグメント名"]))
    )
    work_df = df.loc[mask].copy()

    if work_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot_df = pd.pivot_table(
        work_df,
        index=["指標No", "指標名", "セグメントNo", "セグメント名"],
        columns="調査年度",
        values="指標",
        aggfunc="first",
    ).reset_index()

    if selected_year not in pivot_df.columns or previous_year not in pivot_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    diff_col = get_diff_col_name(selected_year, previous_year)
    pivot_df[diff_col] = (pivot_df[selected_year] - pivot_df[previous_year]).round(1)
    result_df = pivot_df[["指標No", "指標名", "セグメントNo", "セグメント名", previous_year, selected_year, diff_col]].copy()

    positive_df = result_df.sort_values([diff_col, "指標No", "セグメントNo"], ascending=[False, True, True]).head(top_n).reset_index(drop=True)
    negative_df = result_df.sort_values([diff_col, "指標No", "セグメントNo"], ascending=[True, True, True]).head(top_n).reset_index(drop=True)
    return positive_df, negative_df


def generate_overview_summary_comment(
    metric_diff_df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
) -> str:
    if metric_diff_df is None or metric_diff_df.empty:
        return "総括に必要な指標差分データがないため、コメントを生成できません。"

    diff_col = get_diff_col_name(selected_year, previous_year)
    top_positive_df, top_negative_df = build_overview_top_metric_tables(
        metric_diff_df,
        selected_year=selected_year,
        previous_year=previous_year,
        top_n=3,
    )

    if CLIENT is None:
        pos_names = ", ".join(top_positive_df["指標名"].astype(str).tolist()) if not top_positive_df.empty else "該当なし"
        neg_names = ", ".join(top_negative_df["指標名"].astype(str).tolist()) if not top_negative_df.empty else "該当なし"
        return (
            f"{selected_facility}の{selected_year}年度は、前年度比で指標の動きに濃淡がみられます。"
            f" 改善側では{pos_names}が主な押し上げ要因で、悪化側では{neg_names}の確認優先度が高いです。"
            f" まずは全体で大きく動いた指標から確認し、その後セグメント別の寄与を見ていくのが有効です。"
        )

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
施設別KPI前年差の一覧を読み、対象施設の年間総括コメントを日本語で簡潔に作成してください。

要件:
- 3〜5文
- 改善した指標群と悪化した指標群の両方に触れる
- 施設全体として何が起きたかが伝わる表現にする
- 次にどの観点を確認すべきかが自然に伝わる内容にする
- 不明なことは断定しない
"""
    user_prompt = f"""
対象施設: {selected_facility}
対象年度: {selected_year}
比較年度: {previous_year}

指標前年差一覧:
{format_df_for_prompt(metric_diff_df.sort_values(diff_col, ascending=False), max_rows=30)}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=450,
    )
    return response.choices[0].message.content.strip()


def generate_overview_navigation_comment(
    summary_comment: str,
    top_positive_df: pd.DataFrame,
    top_negative_df: pd.DataFrame,
    segment_positive_df: pd.DataFrame,
    segment_negative_df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
) -> str:
    if CLIENT is None:
        return (
            "まずはブロック2で前年差の大きい指標を確認し、改善・悪化の中心がどこかを押さえてください。"
            " その後、ブロック3でどのセグメントが変化を押し上げたか、または押し下げたかを確認するのが有効です。"
        )

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
施設KPIの年間総括を見た利用者に対し、次に確認するとよいポイントを簡潔に案内してください。

要件:
- 2〜3文
- まずブロック2、次にブロック3のように自然な確認順序が伝わるようにする
- 冗長にしない
- 画面上のナビゲーションとして自然な日本語にする
"""
    user_prompt = f"""
対象施設: {selected_facility}
対象年度: {selected_year}
比較年度: {previous_year}

【総括コメント】
{summary_comment}

【改善上位指標】
{format_df_for_prompt(top_positive_df, max_rows=5)}

【悪化上位指標】
{format_df_for_prompt(top_negative_df, max_rows=5)}

【セグメント前年差プラス上位】
{format_df_for_prompt(segment_positive_df, max_rows=5)}

【セグメント前年差マイナス上位】
{format_df_for_prompt(segment_negative_df, max_rows=5)}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


def generate_overview_block2_summary_comment(
    top_positive_df: pd.DataFrame,
    top_negative_df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
) -> str:
    if CLIENT is None:
        pos_names = ", ".join(top_positive_df["指標名"].astype(str).tolist()) if not top_positive_df.empty else "該当なし"
        neg_names = ", ".join(top_negative_df["指標名"].astype(str).tolist()) if not top_negative_df.empty else "該当なし"
        return (
            f"{selected_facility}では、改善側で{pos_names}、悪化側で{neg_names}が大きく動いています。"
            " 今年度の変化を捉えるうえでは、これらの指標を優先して確認するのが有効です。"
        )

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
改善上位指標と悪化上位指標を読み、どの指標が動いたかを簡潔に要約してください。

要件:
- 2〜4文
- 改善と悪化の両方に触れる
- 数値の羅列にせず、何に注目すべきかが伝わる文章にする
- 不明なことは断定しない
"""
    user_prompt = f"""
対象施設: {selected_facility}
対象年度: {selected_year}
比較年度: {previous_year}

【変化量がプラスに大きい指標 Top3】
{format_df_for_prompt(top_positive_df, max_rows=5)}

【変化量がマイナスに大きい指標 Top3】
{format_df_for_prompt(top_negative_df, max_rows=5)}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def generate_overview_block2_navigation_comment(
    block2_summary_comment: str,
    segment_positive_df: pd.DataFrame,
    segment_negative_df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
) -> str:
    if CLIENT is None:
        return "次はブロック3で、前年差の大きい指標がどのセグメントで起きているかを確認してください。"

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
『どの指標が動いたか』を確認した利用者に対し、次にどこを見るべきかを簡潔に案内してください。

要件:
- 2〜3文
- ブロック3で誰が動いたかを見る理由がわかるようにする
- 冗長にしない
"""
    user_prompt = f"""
対象施設: {selected_facility}
対象年度: {selected_year}
比較年度: {previous_year}

【ブロック2サマリー】
{block2_summary_comment}

【セグメント前年差プラス上位】
{format_df_for_prompt(segment_positive_df, max_rows=5)}

【セグメント前年差マイナス上位】
{format_df_for_prompt(segment_negative_df, max_rows=5)}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


def generate_overview_block3_summary_comment(
    segment_positive_df: pd.DataFrame,
    segment_negative_df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
) -> str:
    if CLIENT is None:
        pos_text = ", ".join(segment_positive_df["セグメント名"].astype(str).tolist()) if not segment_positive_df.empty else "該当なし"
        neg_text = ", ".join(segment_negative_df["セグメント名"].astype(str).tolist()) if not segment_negative_df.empty else "該当なし"
        return (
            f"{selected_facility}では、押し上げ側で{pos_text}、押し下げ側で{neg_text}の動きが目立ちます。"
            " 指標ごとの変化がどの層で生じているかを確認すると、全体変化の背景を整理しやすくなります。"
        )

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
セグメント前年差の上位・下位を読み、誰が動いたかを簡潔に要約してください。

要件:
- 2〜4文
- 押し上げたセグメントと押し下げたセグメントの両方に触れる
- 指標とセグメントの組み合わせから注目点がわかるようにする
- 不明なことは断定しない
"""
    user_prompt = f"""
対象施設: {selected_facility}
対象年度: {selected_year}
比較年度: {previous_year}

【セグメント差がプラスに大きい指標】
{format_df_for_prompt(segment_positive_df, max_rows=5)}

【セグメント差がマイナスに大きい指標】
{format_df_for_prompt(segment_negative_df, max_rows=5)}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def generate_overview_block3_navigation_comment(
    block3_summary_comment: str,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
) -> str:
    if CLIENT is None:
        return "重要な示唆は『報告用に使う』で右ペインに残し、必要に応じてKPI変化モードで個別指標の深掘りに進んでください。"

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
『誰が動いたか』を確認した利用者に対し、次に確認するとよいポイントを簡潔に案内してください。

要件:
- 2〜3文
- 重要な示唆を報告用に残すこと、必要に応じて個別深掘りに進むことを自然に案内する
- 冗長にしない
"""
    user_prompt = f"""
対象施設: {selected_facility}
対象年度: {selected_year}
比較年度: {previous_year}

【ブロック3サマリー】
{block3_summary_comment}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


# =========================
# LLMコメント生成
# =========================
def generate_block2_segment_comment(
    metric_name: str,
    metric_diff: float,
    segment_name: str,
    segment_diff: float,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
    segment_table: pd.DataFrame,
) -> str:
    if CLIENT is None:
        direction = "上昇" if segment_diff > 0 else "低下" if segment_diff < 0 else "横ばい"
        return (
            f"{selected_facility}の{selected_year}年度では、指標『{metric_name}』において"
            f"『{segment_name}』が前年差{segment_diff:.1f}ptで{direction}しています。"
            f" 全体前年差{metric_diff:.1f}ptとあわせて、このセグメントが変化の一因かを優先的に確認してください。"
        )

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
施設別KPIの前年差とセグメント別前年差を読み、指定されたセグメントの動きを簡潔に解釈してください。

要件:
- 3〜5文
- 指標全体の変化と、対象セグメントの変化の関係に触れる
- そのセグメントが押し上げ/押し下げ要因の候補かを述べる
- 数値の羅列で終わらせない
- 不明なことは断定しない
"""
    user_prompt = f"""
対象施設: {selected_facility}
対象年度: {selected_year}
比較年度: {previous_year}
対象指標: {metric_name}
対象指標の全体前年差: {metric_diff}
対象セグメント: {segment_name}
対象セグメント前年差: {segment_diff}

同一指標のセグメント別前年差一覧:
{format_df_for_prompt(segment_table, max_rows=30)}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=450,
    )
    return response.choices[0].message.content.strip()



def generate_block3_facility_comment(
    metric_name: str,
    selected_facility: str,
    selected_facility_diff: float,
    comparison_facility: str,
    comparison_facility_diff: float,
    selected_year: str,
    previous_year: str,
    facility_table: pd.DataFrame,
) -> str:
    if CLIENT is None:
        gap = comparison_facility_diff - selected_facility_diff
        relation = "上振れ" if gap > 0 else "下振れ" if gap < 0 else "同水準"
        return (
            f"指標『{metric_name}』では、{comparison_facility}の前年差は{comparison_facility_diff:.1f}ptで、"
            f"{selected_facility}の{selected_facility_diff:.1f}ptに対して{relation}しています。"
            f" 他施設比較の観点では、{comparison_facility}で生じている要因や施策差分を優先的に確認するのが有効です。"
        )

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
施設別KPI前年差を読み、指定施設と比較施設の差を簡潔に解釈してください。

要件:
- 3〜5文
- 選択施設と比較施設の差分の違いに触れる
- どちらが相対的に強い/弱いかを述べる
- 施設間で要因差がありそうかを示唆する
- 不明なことは断定しない
"""
    user_prompt = f"""
対象年度: {selected_year}
比較年度: {previous_year}
対象指標: {metric_name}

選択施設: {selected_facility}
選択施設の前年差: {selected_facility_diff}

比較施設: {comparison_facility}
比較施設の前年差: {comparison_facility_diff}

同一指標の施設別前年差一覧:
{format_df_for_prompt(facility_table, max_rows=50)}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=450,
    )
    return response.choices[0].message.content.strip()



def generate_block1_comment(
    block1_df: pd.DataFrame,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
) -> str:
    if block1_df is None or block1_df.empty:
        return "指標差分データがないため、コメントを生成できません。"

    if CLIENT is None:
        top_up = block1_df.sort_values(f"前年差（{selected_year}-{previous_year}）", ascending=False).head(2)
        top_down = block1_df.sort_values(f"前年差（{selected_year}-{previous_year}）", ascending=True).head(2)
        return (
            f"{selected_facility}の{selected_year}年度は、前年度比で動いた指標の確認が必要です。"
            f" 上昇側では{', '.join(top_up['指標名'].astype(str).tolist())}、"
            f" 低下側では{', '.join(top_down['指標名'].astype(str).tolist())}が主な確認対象です。"
        )

    prompt_df = block1_df.sort_values(f"前年差（{selected_year}-{previous_year}）", ascending=False)
    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
施設別KPI前年差の一覧を読み、どの指標が特に動いたかを日本語で簡潔にまとめてください。

要件:
- 3〜5文
- 上昇した指標と低下した指標の両方に触れる
- 数値の羅列ではなく、確認優先度が伝わる文章にする
- 不明なことは断定しない
"""
    user_prompt = f"""
対象施設: {selected_facility}
対象年度: {selected_year}
比較年度: {previous_year}

指標前年差一覧:
{format_df_for_prompt(prompt_df, max_rows=30)}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()



def generate_segment_trend_comment(
    segment_name: str,
    selected_year: str,
    previous_year: str,
    selected_facility: str,
    segment_diff_row: pd.Series,
) -> str:
    analysis_items = []
    for col in segment_diff_row.index:
        if col in ["セグメントNo", "セグメント名"]:
            continue
        value = segment_diff_row[col]
        if pd.notna(value):
            analysis_items.append(f"{col}: {value}")

    analysis_text = "\n".join(analysis_items) if analysis_items else "差分データなし"

    if CLIENT is None:
        return (
            f"{selected_facility}のセグメント『{segment_name}』について、"
            f"{selected_year}年度と{previous_year}年度の前年差をみると、"
            f"主要指標の上昇・低下が混在しています。"
            f" 特に変化幅の大きい指標を優先して確認し、このセグメント特有の傾向かを見てください。"
        )

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
特定セグメントについて、各指標の前年差一覧を読み、セグメント全体の傾向を簡潔に解釈してください。

要件:
- 3〜5文
- 上昇傾向の指標群と低下傾向の指標群があれば触れる
- そのセグメントの特徴的な変化を簡潔に要約する
- 数値の羅列で終わらせない
- 不明なことは断定しない
"""

    user_prompt = f"""
対象施設: {selected_facility}
対象年度: {selected_year}
比較年度: {previous_year}
対象セグメント: {segment_name}

各指標の前年差一覧:
{analysis_text}
"""

    response = CLIENT.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=450,
    )
    return response.choices[0].message.content.strip()


# =========================
# ピン留め
# =========================
def pin_current_comment() -> None:
    current = st.session_state.current_comment_card
    if not current:
        return

    st.session_state.pinned_comments.append(
        {
            "id": str(uuid.uuid4()),
            **current,
        }
    )


# =========================
# 左ペイン
# =========================
with st.sidebar:
    st.title("施設KPI分析ツール")
    st.markdown("### モード選択")

    button_type_overview = "primary" if st.session_state.mode == "総括" else "secondary"
    if st.button("総括", use_container_width=True, type=button_type_overview):
        set_mode("総括")

    button_type_kpi = "primary" if st.session_state.mode == "KPI変化" else "secondary"
    if st.button("KPI変化", use_container_width=True, type=button_type_kpi):
        set_mode("KPI変化")

    button_type_segment = "primary" if st.session_state.mode == "セグメント分析" else "secondary"
    if st.button("セグメント分析", use_container_width=True, type=button_type_segment):
        set_mode("セグメント分析")

    st.markdown("---")
    st.markdown("### データ読み込み")
    uploaded_file = st.file_uploader(
        "CSVまたはExcelファイルをアップロード",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.success(f"読み込みファイル: {uploaded_file.name}")


if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        reset_analysis_states()
        st.session_state.uploaded_file_name = uploaded_file.name


# =========================
# 画面本体
# =========================
st.title("施設KPI分析ダッシュボード")
st.caption("左ペインでモード選択、中央で分析、右でコメントのピン留め管理を行います。")

main_col, pin_col = st.columns([4.2, 1.5], gap="large")


# =========================
# 右ペイン
# =========================
with pin_col:
    st.markdown("### ピン留めコメント")

    st.markdown("---")

    if not st.session_state.pinned_comments:
        st.info("ピン留めしたコメントはここに表示されます。")
    else:
        for item in st.session_state.pinned_comments:
            with st.container(border=True):
                st.write(f"**種類**: {item.get('type', '-')}")
                st.write(f"**タイトル**: {item.get('title', '-')}")
                st.write(item.get("comment", ""))
                if st.button("削除", key=f"delete_pin_{item['id']}", use_container_width=True):
                    st.session_state.pinned_comments = [
                        x for x in st.session_state.pinned_comments if x["id"] != item["id"]
                    ]
                    st.rerun()

# =========================
# 中央ペイン
# =========================
with main_col:
    with st.container(border=True):
        st.markdown("### 結果表示エリア")

        if uploaded_file is None:
            st.info("左ペインからファイルをアップロードしてください。")
        else:
            try:
                if st.session_state.raw_df is None:
                    st.session_state.raw_df = load_data(uploaded_file)

                raw_df = st.session_state.raw_df.copy()

                if st.session_state.mode not in ["総括", "KPI変化", "セグメント分析"]:
                    st.info("左ペインで『総括』『KPI変化』または『セグメント分析』を押すと分析を表示します。")
                else:
                    year_options = get_year_options(raw_df)
                    facility_options = get_facility_options(raw_df)

                    if not year_options:
                        st.warning("調査年度の選択肢がありません。")
                    elif not facility_options:
                        st.warning("対象施設名の選択肢がありません。")
                    else:
                        st.markdown("### 分析条件")

                        if st.session_state.draft_selected_year is None and year_options:
                            st.session_state.draft_selected_year = year_options[-1]

                        if st.session_state.draft_selected_facility is None and facility_options:
                            st.session_state.draft_selected_facility = facility_options[0]

                        if st.session_state.active_selected_year is None and year_options:
                            st.session_state.active_selected_year = st.session_state.draft_selected_year

                        if st.session_state.active_selected_facility is None and facility_options:
                            st.session_state.active_selected_facility = st.session_state.draft_selected_facility

                        cond_col1, cond_col2, cond_col3 = st.columns([2, 2, 1])

                        with cond_col1:
                            draft_selected_year = st.selectbox(
                                "分析対象｜調査年度",
                                options=year_options,
                                index=year_options.index(st.session_state.draft_selected_year),
                                key="draft_selected_year_widget",
                            )
                            st.session_state.draft_selected_year = draft_selected_year

                        with cond_col2:
                            draft_selected_facility = st.selectbox(
                                "分析対象｜対象施設名",
                                options=facility_options,
                                index=facility_options.index(st.session_state.draft_selected_facility),
                                key="draft_selected_facility_widget",
                            )
                            st.session_state.draft_selected_facility = draft_selected_facility

                        with cond_col3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("再分析", type="primary", use_container_width=True):
                                st.session_state.active_selected_year = st.session_state.draft_selected_year
                                st.session_state.active_selected_facility = st.session_state.draft_selected_facility

                                st.session_state.selected_metric_for_segment = None
                                st.session_state.selected_metric_for_facility = None
                                st.session_state.selected_segment_for_comment = None
                                st.session_state.selected_facility_for_comment = None
                                st.session_state.selected_segment_for_trend = None
                                st.session_state.block1_comment = None
                                st.session_state.block2_generated_comment = None
                                st.session_state.block3_generated_comment = None
                                st.session_state.segment_mode_generated_comment = None
                                st.session_state.overview_summary_comment = None
                                st.session_state.overview_navigation_comment = None
                                st.session_state.overview_block2_summary_comment = None
                                st.session_state.overview_block2_navigation_comment = None
                                st.session_state.overview_block3_summary_comment = None
                                st.session_state.overview_block3_navigation_comment = None
                                st.session_state.current_comment_card = None

                                st.rerun()

                        selected_year = st.session_state.active_selected_year
                        selected_facility = st.session_state.active_selected_facility

                        st.caption(f"現在表示中: {selected_year} / {selected_facility}")

                        if (
                            st.session_state.draft_selected_year != st.session_state.active_selected_year
                            or st.session_state.draft_selected_facility != st.session_state.active_selected_facility
                        ):
                            st.info(
                                f"選択中の条件は未反映です。"
                                f" 再分析を押すと {st.session_state.draft_selected_year} / "
                                f"{st.session_state.draft_selected_facility} に切り替わります。"
                            )

                        previous_year = get_previous_year(selected_year, year_options)

                        if previous_year is None:
                            st.warning("比較対象となる前年度データがありません。")

                        elif st.session_state.mode == "総括":
                            diff_col = get_diff_col_name(selected_year, previous_year)
                            st.caption(f"比較対象: {selected_facility} / {selected_year} vs {previous_year}")

                            metric_diff_df = build_block1_metric_diff_table(
                                raw_df,
                                selected_year=selected_year,
                                previous_year=previous_year,
                                selected_facility=selected_facility,
                            )
                            top_positive_df, top_negative_df = build_overview_top_metric_tables(
                                metric_diff_df,
                                selected_year=selected_year,
                                previous_year=previous_year,
                                top_n=3,
                            )
                            segment_positive_df, segment_negative_df = build_overview_segment_summary_tables(
                                raw_df,
                                selected_year=selected_year,
                                previous_year=previous_year,
                                selected_facility=selected_facility,
                                top_n=3,
                            )

                            if metric_diff_df.empty:
                                st.info("総括に必要な指標差分データがありません。")
                            else:
                                if not st.session_state.overview_summary_comment:
                                    st.session_state.overview_summary_comment = generate_overview_summary_comment(
                                        metric_diff_df=metric_diff_df,
                                        selected_year=selected_year,
                                        previous_year=previous_year,
                                        selected_facility=selected_facility,
                                    )

                                if not st.session_state.overview_navigation_comment:
                                    st.session_state.overview_navigation_comment = generate_overview_navigation_comment(
                                        summary_comment=st.session_state.overview_summary_comment,
                                        top_positive_df=top_positive_df,
                                        top_negative_df=top_negative_df,
                                        segment_positive_df=segment_positive_df,
                                        segment_negative_df=segment_negative_df,
                                        selected_year=selected_year,
                                        previous_year=previous_year,
                                        selected_facility=selected_facility,
                                    )

                                if not st.session_state.overview_block2_summary_comment:
                                    st.session_state.overview_block2_summary_comment = generate_overview_block2_summary_comment(
                                        top_positive_df=top_positive_df,
                                        top_negative_df=top_negative_df,
                                        selected_year=selected_year,
                                        previous_year=previous_year,
                                        selected_facility=selected_facility,
                                    )

                                if not st.session_state.overview_block2_navigation_comment:
                                    st.session_state.overview_block2_navigation_comment = generate_overview_block2_navigation_comment(
                                        block2_summary_comment=st.session_state.overview_block2_summary_comment,
                                        segment_positive_df=segment_positive_df,
                                        segment_negative_df=segment_negative_df,
                                        selected_year=selected_year,
                                        previous_year=previous_year,
                                        selected_facility=selected_facility,
                                    )

                                if not st.session_state.overview_block3_summary_comment:
                                    st.session_state.overview_block3_summary_comment = generate_overview_block3_summary_comment(
                                        segment_positive_df=segment_positive_df,
                                        segment_negative_df=segment_negative_df,
                                        selected_year=selected_year,
                                        previous_year=previous_year,
                                        selected_facility=selected_facility,
                                    )

                                if not st.session_state.overview_block3_navigation_comment:
                                    st.session_state.overview_block3_navigation_comment = generate_overview_block3_navigation_comment(
                                        block3_summary_comment=st.session_state.overview_block3_summary_comment,
                                        selected_year=selected_year,
                                        previous_year=previous_year,
                                        selected_facility=selected_facility,
                                    )

                                with st.container(border=True):
                                    st.markdown("### ブロック1｜今月は何が起きたか？")
                                    st.markdown("#### 全体の改善 / 悪化サマリー")
                                    st.write(st.session_state.overview_summary_comment)

                                    st.markdown("#### 次に確認するとよいポイント")
                                    st.info(st.session_state.overview_navigation_comment)

                                    if st.button("報告用に使う", key="pin_overview_summary_btn", use_container_width=False):
                                        st.session_state.current_comment_card = {
                                            "type": "総括｜ブロック1",
                                            "title": f"{selected_facility}｜{selected_year}年度｜総括",
                                            "comment": st.session_state.overview_summary_comment,
                                        }
                                        pin_current_comment()
                                        st.success("総括コメントをピン留めしました。")
                                        st.rerun()

                                with st.container(border=True):
                                    st.markdown("### ブロック2｜どの指標が動いたか？")
                                    st.markdown("#### 全体の改善 / 悪化サマリー")
                                    st.write(st.session_state.overview_block2_summary_comment)

                                    st.markdown("#### 次に確認するとよいポイント")
                                    st.info(st.session_state.overview_block2_navigation_comment)

                                    if st.button("報告用に使う", key="pin_overview_block2_summary_btn", use_container_width=False):
                                        st.session_state.current_comment_card = {
                                            "type": "総括｜ブロック2",
                                            "title": f"{selected_facility}｜{selected_year}年度｜どの指標が動いたか",
                                            "comment": st.session_state.overview_block2_summary_comment,
                                        }
                                        pin_current_comment()
                                        st.success("ブロック2コメントをピン留めしました。")
                                        st.rerun()

                                    st.markdown("#### 変化量がプラスに大きい指標 Top3")
                                    render_readable_table(
                                        top_positive_df,
                                        display_columns=["指標No", "指標名", diff_col],
                                        decimal_columns=[diff_col],
                                        font_size_px=16,
                                    )

                                    st.markdown("#### 変化量がマイナスに大きい指標 Top3")
                                    render_readable_table(
                                        top_negative_df,
                                        display_columns=["指標No", "指標名", diff_col],
                                        decimal_columns=[diff_col],
                                        font_size_px=16,
                                    )

                                with st.container(border=True):
                                    st.markdown("### ブロック3｜誰が動いたか？")
                                    st.markdown("#### 全体の改善 / 悪化サマリー")
                                    st.write(st.session_state.overview_block3_summary_comment)

                                    st.markdown("#### 次に確認するとよいポイント")
                                    st.info(st.session_state.overview_block3_navigation_comment)

                                    if st.button("報告用に使う", key="pin_overview_block3_summary_btn", use_container_width=False):
                                        st.session_state.current_comment_card = {
                                            "type": "総括｜ブロック3",
                                            "title": f"{selected_facility}｜{selected_year}年度｜誰が動いたか",
                                            "comment": st.session_state.overview_block3_summary_comment,
                                        }
                                        pin_current_comment()
                                        st.success("ブロック3コメントをピン留めしました。")
                                        st.rerun()

                                    st.markdown("#### セグメント差がプラスに大きい指標")
                                    render_readable_table(
                                        segment_positive_df,
                                        display_columns=["指標No", "指標名", "セグメントNo", "セグメント名", diff_col],
                                        decimal_columns=[diff_col],
                                        font_size_px=16,
                                    )

                                    st.markdown("#### セグメント差がマイナスに大きい指標")
                                    render_readable_table(
                                        segment_negative_df,
                                        display_columns=["指標No", "指標名", "セグメントNo", "セグメント名", diff_col],
                                        decimal_columns=[diff_col],
                                        font_size_px=16,
                                    )

                        elif st.session_state.mode == "KPI変化":
                            diff_col = f"前年差（{selected_year}-{previous_year}）"
                            st.caption(f"比較対象: {selected_facility} / {selected_year} vs {previous_year}")

                            block1_df = build_block1_metric_diff_table(
                                raw_df,
                                selected_year=selected_year,
                                previous_year=previous_year,
                                selected_facility=selected_facility,
                            )

                            with st.container(border=True):
                                st.markdown("### ブロック1｜どの指標が動いたか？")

                                if block1_df.empty:
                                    st.info("該当データがありません。")
                                else:
                                    block1_sort_order = st.selectbox(
                                        "並び順",
                                        options=["前年差が大きい順", "前年差が小さい順", "No順"],
                                        index=0,
                                        key=f"block1_sort_{selected_year}_{selected_facility}",
                                    )

                                    if block1_sort_order == "前年差が大きい順":
                                        block1_df = block1_df.sort_values(
                                            by=[diff_col, "指標No", "指標名"],
                                            ascending=[False, True, True],
                                        ).reset_index(drop=True)
                                    elif block1_sort_order == "前年差が小さい順":
                                        block1_df = block1_df.sort_values(
                                            by=[diff_col, "指標No", "指標名"],
                                            ascending=[True, True, True],
                                        ).reset_index(drop=True)
                                    else:
                                        block1_df = block1_df.sort_values(
                                            by=["指標No", "指標名"],
                                            ascending=[True, True],
                                        ).reset_index(drop=True)

                                    header_cols = st.columns([1, 4.5, 2.2, 3.2])
                                    header_cols[0].markdown("**No**")
                                    header_cols[1].markdown("**指標名**")
                                    header_cols[2].markdown(f"**{diff_col}**")
                                    header_cols[3].markdown("**操作**")

                                    for _, row in block1_df.iterrows():
                                        metric_no = int(row["指標No"]) if pd.notna(row["指標No"]) else None
                                        row_cols = st.columns([1, 4.5, 2.2, 3.2])
                                        row_cols[0].write(metric_no)
                                        row_cols[1].write(row["指標名"])
                                        row_cols[2].markdown(format_diff_value_html(row[diff_col]), unsafe_allow_html=True)

                                        btn_col1, btn_col2 = row_cols[3].columns(2)
                                        if btn_col1.button(
                                            "誰が動いた？",
                                            key=f"who_moved_{selected_year}_{selected_facility}_{metric_no}",
                                            use_container_width=True,
                                        ):
                                            st.session_state.selected_metric_for_segment = {
                                                "metric_no": metric_no,
                                                "metric_name": row["指標名"],
                                                "diff_value": row[diff_col],
                                                "selected_year": selected_year,
                                                "previous_year": previous_year,
                                                "facility": selected_facility,
                                            }
                                            st.rerun()

                                        if btn_col2.button(
                                            "他の施設は？",
                                            key=f"other_facility_{selected_year}_{selected_facility}_{metric_no}",
                                            use_container_width=True,
                                        ):
                                            st.session_state.selected_metric_for_facility = {
                                                "metric_no": metric_no,
                                                "metric_name": row["指標名"],
                                                "diff_value": row[diff_col],
                                                "selected_year": selected_year,
                                                "previous_year": previous_year,
                                                "facility": selected_facility,
                                            }
                                            st.rerun()

                                    st.markdown("---")
                                    if st.button("ブロック1コメント生成", key="generate_block1_comment_btn"):
                                        with st.spinner("ブロック1コメントを生成中です..."):
                                            comment = generate_block1_comment(
                                                block1_df=block1_df,
                                                selected_year=selected_year,
                                                previous_year=previous_year,
                                                selected_facility=selected_facility,
                                            )
                                        st.session_state.block1_comment = comment
                                        st.session_state.current_comment_card = {
                                            "type": "ブロック1コメント",
                                            "title": f"{selected_facility}｜{selected_year}年度の指標変化",
                                            "comment": comment,
                                        }
                                        st.rerun()

                                    if st.session_state.block1_comment:
                                        with st.container(border=True):
                                            st.markdown("#### ブロック1コメント")
                                            st.write(st.session_state.block1_comment)

                            with st.container(border=True):
                                st.markdown("### ブロック2｜誰が動いたか？")

                                selected_metric_segment = st.session_state.selected_metric_for_segment
                                if not selected_metric_segment:
                                    st.info("ブロック1の『誰が動いた？』を押すと、ここにセグメント別前年差を表示します。")
                                else:
                                    st.write(
                                        f"対象指標: {selected_metric_segment['metric_no']} - {selected_metric_segment['metric_name']}"
                                    )
                                    seg_df = build_block2_segment_diff_table(
                                        raw_df,
                                        selected_year=selected_year,
                                        previous_year=previous_year,
                                        selected_facility=selected_facility,
                                        metric_no=selected_metric_segment["metric_no"],
                                    )

                                    if seg_df.empty:
                                        st.info("セグメント差分データがありません。")
                                    else:
                                        block2_sort_order = st.selectbox(
                                            "並び順",
                                            options=["前年差が大きい順", "前年差が小さい順", "No順"],
                                            index=0,
                                            key=f"block2_sort_{selected_year}_{selected_facility}_{selected_metric_segment['metric_no']}",
                                        )

                                        if block2_sort_order == "前年差が大きい順":
                                            seg_df = seg_df.sort_values(
                                                by=[diff_col, "セグメントNo", "セグメント名"],
                                                ascending=[False, True, True],
                                            ).reset_index(drop=True)
                                        elif block2_sort_order == "前年差が小さい順":
                                            seg_df = seg_df.sort_values(
                                                by=[diff_col, "セグメントNo", "セグメント名"],
                                                ascending=[True, True, True],
                                            ).reset_index(drop=True)
                                        else:
                                            seg_df = seg_df.sort_values(
                                                by=["セグメントNo", "セグメント名"],
                                                ascending=[True, True],
                                            ).reset_index(drop=True)

                                        header_cols = st.columns([1, 4.5, 2.2, 2.2])
                                        header_cols[0].markdown("**No**")
                                        header_cols[1].markdown("**セグメント名**")
                                        header_cols[2].markdown(f"**{diff_col}**")
                                        header_cols[3].markdown("**操作**")

                                        for _, seg_row in seg_df.iterrows():
                                            segment_no = int(seg_row["セグメントNo"]) if pd.notna(seg_row["セグメントNo"]) else None
                                            seg_cols = st.columns([1, 4.5, 2.2, 2.2])
                                            seg_cols[0].write(segment_no)
                                            seg_cols[1].write(seg_row["セグメント名"])
                                            seg_cols[2].markdown(format_diff_value_html(seg_row[diff_col]), unsafe_allow_html=True)

                                            if seg_cols[3].button(
                                                "コメント生成",
                                                key=f"segment_comment_{selected_year}_{selected_facility}_{selected_metric_segment['metric_no']}_{segment_no}",
                                                use_container_width=True,
                                            ):
                                                with st.spinner("セグメントコメントを生成中です..."):
                                                    comment = generate_block2_segment_comment(
                                                        metric_name=selected_metric_segment["metric_name"],
                                                        metric_diff=float(selected_metric_segment["diff_value"]),
                                                        segment_name=str(seg_row["セグメント名"]),
                                                        segment_diff=float(seg_row[diff_col]),
                                                        selected_year=selected_year,
                                                        previous_year=previous_year,
                                                        selected_facility=selected_facility,
                                                        segment_table=seg_df,
                                                    )
                                                st.session_state.selected_segment_for_comment = {
                                                    "segment_no": segment_no,
                                                    "segment_name": str(seg_row["セグメント名"]),
                                                    "segment_diff": float(seg_row[diff_col]),
                                                }
                                                st.session_state.block2_generated_comment = comment
                                                st.session_state.current_comment_card = {
                                                    "type": "ブロック2コメント",
                                                    "title": f"{selected_facility}｜{selected_metric_segment['metric_name']}｜{seg_row['セグメント名']}",
                                                    "comment": comment,
                                                }
                                                st.rerun()

                                        st.markdown("---")
                                        st.markdown("#### ブロック2コメント")
                                        if st.session_state.block2_generated_comment:
                                            with st.container(border=True):
                                                selected_seg = st.session_state.selected_segment_for_comment
                                                if selected_seg is not None:
                                                    st.write(f"**対象セグメント**: {selected_seg['segment_no']} - {selected_seg['segment_name']}")
                                                    st.write(f"**セグメント前年差**: {selected_seg['segment_diff']:.1f}")

                                                st.write(st.session_state.block2_generated_comment)

                                                if st.button(
                                                    "このコメントをピン留め",
                                                    key="pin_block2_generated_comment_btn",
                                                    use_container_width=False,
                                                ):
                                                    pin_current_comment()
                                                    st.success("コメントをピン留めしました。")
                                                    st.rerun()
                                        else:
                                            st.info("各セグメントの『コメント生成』を押すと、ここに分析コメントを表示します。")

                            with st.container(border=True):
                                st.markdown("### ブロック3｜ほかの施設はどうか？")

                                selected_metric_facility = st.session_state.selected_metric_for_facility
                                if not selected_metric_facility:
                                    st.info("ブロック1の『他の施設は？』を押すと、ここに施設別前年差を表示します。")
                                else:
                                    st.write(
                                        f"対象指標: {selected_metric_facility['metric_no']} - {selected_metric_facility['metric_name']}"
                                    )
                                    facility_df = build_block3_facility_diff_table(
                                        raw_df,
                                        selected_year=selected_year,
                                        previous_year=previous_year,
                                        metric_no=selected_metric_facility["metric_no"],
                                    )

                                    if facility_df.empty:
                                        st.info("施設比較データがありません。")
                                    else:
                                        block3_sort_order = st.selectbox(
                                            "並び順",
                                            options=["前年差が大きい順", "前年差が小さい順", "施設名順"],
                                            index=0,
                                            key=f"block3_sort_{selected_year}_{selected_metric_facility['metric_no']}",
                                        )

                                        if block3_sort_order == "前年差が大きい順":
                                            facility_df = facility_df.sort_values(
                                                by=[diff_col, "対象施設名"],
                                                ascending=[False, True],
                                            ).reset_index(drop=True)
                                        elif block3_sort_order == "前年差が小さい順":
                                            facility_df = facility_df.sort_values(
                                                by=[diff_col, "対象施設名"],
                                                ascending=[True, True],
                                            ).reset_index(drop=True)
                                        else:
                                            facility_df = facility_df.sort_values(
                                                by=["対象施設名"],
                                                ascending=[True],
                                            ).reset_index(drop=True)

                                        selected_facility_row = facility_df[
                                            facility_df["対象施設名"] == selected_facility
                                        ]
                                        selected_facility_diff = None
                                        if not selected_facility_row.empty:
                                            selected_facility_diff = float(selected_facility_row.iloc[0][diff_col])

                                        header_cols = st.columns([4.4, 2.2, 2.0])
                                        header_cols[0].markdown("**対象施設名**")
                                        header_cols[1].markdown(f"**{diff_col}**")
                                        header_cols[2].markdown("**操作**")

                                        for _, facility_row in facility_df.iterrows():
                                            facility_name = str(facility_row["対象施設名"])
                                            row_cols = st.columns([4.4, 2.2, 2.0])
                                            display_name = f"{facility_name} ★" if facility_name == selected_facility else facility_name
                                            row_cols[0].write(display_name)
                                            row_cols[1].markdown(format_diff_value_html(facility_row[diff_col]), unsafe_allow_html=True)

                                            if facility_name == selected_facility:
                                                row_cols[2].write("-")
                                            else:
                                                if row_cols[2].button(
                                                    "コメント生成",
                                                    key=f"facility_comment_{selected_year}_{selected_metric_facility['metric_no']}_{facility_name}",
                                                    use_container_width=True,
                                                ):
                                                    with st.spinner("施設比較コメントを生成中です..."):
                                                        comment = generate_block3_facility_comment(
                                                            metric_name=selected_metric_facility["metric_name"],
                                                            selected_facility=selected_facility,
                                                            selected_facility_diff=float(selected_facility_diff) if selected_facility_diff is not None else 0.0,
                                                            comparison_facility=facility_name,
                                                            comparison_facility_diff=float(facility_row[diff_col]),
                                                            selected_year=selected_year,
                                                            previous_year=previous_year,
                                                            facility_table=facility_df,
                                                        )
                                                    st.session_state.selected_facility_for_comment = {
                                                        "facility_name": facility_name,
                                                        "facility_diff": float(facility_row[diff_col]),
                                                        "selected_facility": selected_facility,
                                                        "selected_facility_diff": float(selected_facility_diff) if selected_facility_diff is not None else 0.0,
                                                    }
                                                    st.session_state.block3_generated_comment = comment
                                                    st.session_state.current_comment_card = {
                                                        "type": "ブロック3コメント",
                                                        "title": f"{selected_metric_facility['metric_name']}｜{selected_facility} vs {facility_name}",
                                                        "comment": comment,
                                                    }
                                                    st.rerun()

                                        st.markdown("---")
                                        st.markdown("#### ブロック3コメント")
                                        if st.session_state.block3_generated_comment:
                                            with st.container(border=True):
                                                selected_fac = st.session_state.selected_facility_for_comment
                                                if selected_fac is not None:
                                                    st.write(f"**選択施設**: {selected_fac['selected_facility']} ({selected_fac['selected_facility_diff']:.1f})")
                                                    st.write(f"**比較施設**: {selected_fac['facility_name']} ({selected_fac['facility_diff']:.1f})")

                                                st.write(st.session_state.block3_generated_comment)

                                                if st.button(
                                                    "このコメントをピン留め",
                                                    key="pin_block3_generated_comment_btn",
                                                    use_container_width=False,
                                                ):
                                                    pin_current_comment()
                                                    st.success("コメントをピン留めしました。")
                                                    st.rerun()
                                        else:
                                            st.info("各施設の『コメント生成』を押すと、ここに施設比較コメントを表示します。")

                        elif st.session_state.mode == "セグメント分析":
                            st.caption(f"比較対象: {selected_facility} / {selected_year} vs {previous_year}")

                            segment_analysis_df = build_segment_analysis_diff_table(
                                raw_df,
                                selected_year=selected_year,
                                previous_year=previous_year,
                                selected_facility=selected_facility,
                            )

                            with st.container(border=True):
                                st.markdown("### セグメント分析｜セグメントごとの指標前年差")

                                if segment_analysis_df.empty:
                                    st.info("該当データがありません。")
                                else:
                                    metric_cols = [
                                        c for c in segment_analysis_df.columns
                                        if c not in ["セグメントNo", "セグメント名"]
                                    ]

                                    for _, seg_row in segment_analysis_df.iterrows():
                                        segment_no = int(seg_row["セグメントNo"]) if pd.notna(seg_row["セグメントNo"]) else None

                                        left_col, center_col, right_col = st.columns([0.9, 2.6, 8.5])

                                        with left_col:
                                            st.markdown("**No**")
                                            st.markdown(
                                                f"<div style='font-size:16px;'>{segment_no if segment_no is not None else '-'}</div>",
                                                unsafe_allow_html=True,
                                            )

                                        with center_col:
                                            st.markdown("**セグメント / 操作**")
                                            st.markdown(
                                                f"<div style='font-size:16px; margin-bottom:8px;'>{seg_row['セグメント名']}</div>",
                                                unsafe_allow_html=True,
                                            )

                                            if st.button(
                                                "傾向分析",
                                                key=f"segment_trend_{selected_year}_{selected_facility}_{segment_no}",
                                                use_container_width=True,
                                            ):
                                                with st.spinner("セグメント傾向を分析中です..."):
                                                    comment = generate_segment_trend_comment(
                                                        segment_name=str(seg_row["セグメント名"]),
                                                        selected_year=selected_year,
                                                        previous_year=previous_year,
                                                        selected_facility=selected_facility,
                                                        segment_diff_row=seg_row,
                                                    )

                                                st.session_state.selected_segment_for_trend = {
                                                    "segment_no": segment_no,
                                                    "segment_name": str(seg_row["セグメント名"]),
                                                }
                                                st.session_state.segment_mode_generated_comment = comment
                                                st.session_state.current_comment_card = {
                                                    "type": "セグメント分析コメント",
                                                    "title": f"{selected_facility}｜{seg_row['セグメント名']}｜セグメント傾向",
                                                    "comment": comment,
                                                }
                                                st.rerun()

                                        with right_col:
                                            st.markdown("**指標前年差一覧**")
                                            metrics_html = build_segment_metrics_html_table(metric_cols, seg_row)
                                            st.markdown(metrics_html, unsafe_allow_html=True)

                                        st.markdown("---")

                                    st.markdown("#### セグメント傾向の分析結果")

                                    if st.session_state.segment_mode_generated_comment:
                                        with st.container(border=True):
                                            selected_seg = st.session_state.selected_segment_for_trend
                                            if selected_seg is not None:
                                                st.write(f"**対象セグメント**: {selected_seg['segment_no']} - {selected_seg['segment_name']}")
                                            st.write(st.session_state.segment_mode_generated_comment)

                                            if st.button(
                                                "この結果をピン留め",
                                                key="pin_segment_mode_comment_btn",
                                                use_container_width=False,
                                            ):
                                                pin_current_comment()
                                                st.success("コメントをピン留めしました。")
                                                st.rerun()
                                    else:
                                        st.info("各セグメントの『傾向分析』を押すと、ここに分析結果を表示します。")

            except Exception as e:
                st.error(f"表示中にエラーが発生しました: {e}")


st.markdown("---")
st.write("KPI変化モードとセグメント分析モードを切り替えて、中央ペインで分析、右ペインでコメントのピン留め管理を行えます。")
