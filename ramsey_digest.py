import os
import re
import time
import arxiv
import requests
import smtplib
import hashlib
import traceback
from email.message import EmailMessage
from email.utils import formataddr
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta, timezone

# ============================================================
#  统一论文数据结构
# ============================================================
@dataclass
class Paper:
    title: str
    authors: List[str]
    abstract: str
    url: str
    source: str
    published: str
    doi: Optional[str] = None
    journal: Optional[str] = None

    @property
    def fingerprint(self):
        normalized = re.sub(r'[^a-z0-9]', '', self.title.lower())
        return hashlib.md5(normalized.encode()).hexdigest()


# ============================================================
#  配置
# ============================================================
now_utc = datetime.now(timezone.utc)
since_utc = now_utc - timedelta(days=1)
DATE_STR = now_utc.strftime("%Y-%m-%d")

RAMSEY_KEYWORDS = [
    "Ramsey", "Gallai-Ramsey", "Ramsey number", "Ramsey multiplicity",
    "size Ramsey", "anti-Ramsey", "monochromatic subgraph",
    "Schur number", "Rado theorem", "Hales-Jewett",
    "Ramsey-type", "graph coloring Ramsey",
]

KEYWORD_PATTERN = re.compile(
    '|'.join(re.escape(kw) for kw in RAMSEY_KEYWORDS),
    re.IGNORECASE
)


def is_ramsey_related(title, abstract=""):
    text = f"{title} {abstract}"
    return bool(KEYWORD_PATTERN.search(text))


# ============================================================
#  Source 1: arXiv
# ============================================================
def fetch_arxiv() -> List[Paper]:
    print(">>> [arXiv] querying...")

    query = (
        'cat:math.CO AND ('
        'all:Ramsey OR all:"Gallai-Ramsey" OR all:"Ramsey number" '
        'OR all:"size Ramsey" OR all:"monochromatic subgraph" '
        'OR all:"Ramsey multiplicity" OR all:"anti-Ramsey"'
        ')'
    )

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=80,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    try:
        for r in client.results(search):
            if r.published.replace(tzinfo=timezone.utc) < since_utc:
                continue
            papers.append(Paper(
                title=r.title,
                authors=[a.name for a in r.authors],
                abstract=r.summary.strip(),
                url=r.entry_id,
                source="arXiv",
                published=r.published.strftime("%Y-%m-%d"),
                doi=r.doi,
            ))
    except Exception as e:
        print(f"  error: {e}")

    print(f"  arXiv: {len(papers)} papers")
    return papers


# ============================================================
#  Source 2: Semantic Scholar
# ============================================================
def fetch_semantic_scholar() -> List[Paper]:
    print(">>> [Semantic Scholar] querying...")

    S2_API = "https://api.semanticscholar.org/graph/v1"
    papers = []

    search_queries = ["Ramsey theory graph", "Ramsey number combinatorics"]

    for sq in search_queries:
        try:
            url = f"{S2_API}/paper/search"
            params = {
                "query": sq,
                "year": str(now_utc.year),
                "fieldsOfStudy": "Mathematics",
                "fields": "title,authors,abstract,url,externalIds,publicationDate,venue",
                "limit": 30,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                pub_date = item.get("publicationDate", "")
                if pub_date:
                    try:
                        pd = datetime.strptime(pub_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        if pd < since_utc - timedelta(days=7):
                            continue
                    except ValueError:
                        pass

                title = item.get("title", "")
                abstract = item.get("abstract", "") or ""

                if not is_ramsey_related(title, abstract):
                    continue

                authors = [a.get("name", "") for a in item.get("authors", [])]
                ext_ids = item.get("externalIds", {}) or {}

                papers.append(Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=item.get("url", ""),
                    source="Semantic Scholar",
                    published=pub_date or "unknown",
                    doi=ext_ids.get("DOI"),
                    journal=item.get("venue", ""),
                ))
            time.sleep(1)
        except Exception as e:
            print(f"  S2 query '{sq}' failed: {e}")

    print(f"  Semantic Scholar: {len(papers)} papers")
    return papers


# ============================================================
#  Source 3: OpenAlex
# ============================================================
def fetch_openalex() -> List[Paper]:
    print(">>> [OpenAlex] querying...")

    papers = []
    try:
        url = "https://api.openalex.org/works"
        params = {
            "search": "Ramsey theory graph",
            "filter": (
                f"from_publication_date:{since_utc.strftime('%Y-%m-%d')},"
                f"type:article|preprint"
            ),
            "sort": "publication_date:desc",
            "per_page": 30,
            "mailto": os.environ.get("SENDER_EMAIL", "test@example.com"),
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for work in data.get("results", []):
            title = work.get("title", "")
            abstract_inv = work.get("abstract_inverted_index", {})

            if abstract_inv:
                max_pos = max(pos for positions in abstract_inv.values() for pos in positions)
                abstract_words = [""] * (max_pos + 1)
                for word, positions in abstract_inv.items():
                    for pos in positions:
                        abstract_words[pos] = word
                abstract = " ".join(abstract_words)
            else:
                abstract = ""

            if not is_ramsey_related(title, abstract):
                continue

            authors = []
            for authorship in work.get("authorships", [])[:5]:
                author_name = authorship.get("author", {}).get("display_name", "")
                if author_name:
                    authors.append(author_name)

            doi_url = work.get("doi", "")
            doi = doi_url.replace("https://doi.org/", "") if doi_url else None

            primary_loc = work.get("primary_location") or {}

            papers.append(Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                url=primary_loc.get("landing_page_url", "") or doi_url or "",
                source="OpenAlex",
                published=work.get("publication_date", "unknown"),
                doi=doi,
                journal=primary_loc.get("source", {}).get("display_name", "") if primary_loc else "",
            ))
    except Exception as e:
        print(f"  OpenAlex error: {e}")
        traceback.print_exc()

    print(f"  OpenAlex: {len(papers)} papers")
    return papers


# ============================================================
#  Source 4: Journal RSS
# ============================================================
def fetch_journal_rss() -> List[Paper]:
    print(">>> [Journal RSS] querying...")

    RSS_FEEDS = {
        "J. Combin. Theory B": "https://rss.sciencedirect.com/publication/science/00958956",
        "Discrete Math": "https://rss.sciencedirect.com/publication/science/0012365X",
        "European J. Combin.": "https://rss.sciencedirect.com/publication/science/01956698",
        "Electron. J. Combin.": "https://www.combinatorics.org/ojs/index.php/eljc/gateway/plugin/WebFeedGatewayPlugin/atom",
    }

    papers = []

    try:
        import feedparser
    except ImportError:
        print("  feedparser not installed, skipping")
        return papers

    for journal_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                summary_clean = re.sub(r"<[^>]+>", "", summary).strip()

                if not is_ramsey_related(title, summary_clean):
                    continue

                pub_date = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = time.strftime("%Y-%m-%d", entry.published_parsed)

                authors = []
                if hasattr(entry, "authors"):
                    authors = [a.get("name", "") for a in entry.authors]
                elif hasattr(entry, "author"):
                    authors = [entry.author]

                papers.append(Paper(
                    title=title,
                    authors=authors,
                    abstract=summary_clean[:500],
                    url=entry.get("link", ""),
                    source=f"Journal:{journal_name}",
                    published=pub_date,
                    journal=journal_name,
                ))
        except Exception as e:
            print(f"  {journal_name} RSS failed: {e}")
        time.sleep(0.5)

    print(f"  Journal RSS: {len(papers)} papers")
    return papers


# ============================================================
#  Source 5: MathOverflow
# ============================================================
def fetch_mathoverflow() -> List[Paper]:
    print(">>> [MathOverflow] querying...")

    papers = []
    try:
        url = "https://api.stackexchange.com/2.3/search/advanced"
        params = {
            "order": "desc",
            "sort": "creation",
            "q": "Ramsey",
            "tagged": "combinatorics",
            "site": "mathoverflow",
            "fromdate": int(since_utc.timestamp()),
            "pagesize": 10,
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            title = item.get("title", "")
            body = re.sub(r"<[^>]+>", "", item.get("body", "") if "body" in item else "")[:400]

            if not is_ramsey_related(title, body):
                continue

            owner = item.get("owner", {}).get("display_name", "anonymous")
            creation = datetime.fromtimestamp(
                item["creation_date"], tz=timezone.utc
            ).strftime("%Y-%m-%d")

            papers.append(Paper(
                title=f"[MO] {title}",
                authors=[owner],
                abstract=body,
                url=item.get("link", ""),
                source="MathOverflow",
                published=creation,
            ))
    except Exception as e:
        print(f"  MathOverflow error: {e}")

    print(f"  MathOverflow: {len(papers)} papers")
    return papers


# ============================================================
#  Dedup
# ============================================================
def deduplicate(all_papers: List[Paper]) -> List[Paper]:
    seen = {}
    for paper in all_papers:
        fp = paper.fingerprint
        if fp not in seen:
            seen[fp] = paper
        else:
            existing = seen[fp]
            score_new = (
                (1 if paper.doi else 0)
                + (1 if len(paper.abstract) > 100 else 0)
                + (2 if paper.source == "arXiv" else 0)
                + (1 if paper.journal else 0)
            )
            score_old = (
                (1 if existing.doi else 0)
                + (1 if len(existing.abstract) > 100 else 0)
                + (2 if existing.source == "arXiv" else 0)
                + (1 if existing.journal else 0)
            )
            if score_new > score_old:
                seen[fp] = paper
            if paper.source not in seen[fp].source:
                seen[fp].source += f" + {paper.source}"

    result = sorted(seen.values(), key=lambda p: p.published or "0000", reverse=True)
    print(f"Dedup: {len(all_papers)} -> {len(result)}")
    return result


# ============================================================
#  Build Prompt
# ============================================================
def build_prompt(papers: List[Paper]) -> str:
    papers_text = ""
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:4])
        if len(p.authors) > 4:
            authors += " et al."
        abstract = " ".join(p.abstract.split())

        papers_text += f"""
---
Paper {i}:
Title: {p.title}
Authors: {authors}
Source: {p.source}
Date: {p.published}
Journal: {p.journal or 'preprint'}
Link: {p.url}
DOI: {p.doi or 'N/A'}
Abstract: {abstract}
---
"""

    prompt = f"""你是图论与 Ramsey Theory 方向的数学研究专家，精通中文学术写作。

以下是今日（{DATE_STR}）从多个学术数据源（arXiv、Semantic Scholar、
OpenAlex、核心期刊、MathOverflow 等）汇总的 {len(papers)} 篇与
Ramsey Theory 相关的最新论文/讨论。

## 对每篇论文/讨论，请输出：

### 📄 论文 X：[中文翻译的标题]
- **原标题**：[英文原标题]
- **作者**：[作者]
- **来源**：[数据源 + 期刊名]
- **链接**：[URL]
- **中文摘要**（3-5句，专业且易懂）
- **主要贡献**（核心新结果/新方法）
- **技术方法**（关键工具）
- **学术脉络**（属于 Ramsey Theory 哪个分支，关联哪些经典结果）
- **推荐指数**：⭐-⭐⭐⭐⭐⭐

对于 MathOverflow 讨论，改为分析问题的学术价值和已有回答的要点。

## 最后给出「今日综述」：
1. 今日整体趋势
2. 最值得关注的 1-2 篇
3. 与近期研究方向的关联（如 Gallai-Ramsey 数、图着色 Ramsey 问题等）

论文列表：
{papers_text}

请用 Markdown 格式输出，包含数学公式时使用 $...$ 格式。"""

    return prompt


# ============================================================
#  AI 调用函数
# ============================================================
def call_deepseek(prompt: str) -> str:
    """调用 DeepSeek API（OpenAI 兼容格式）"""
    from openai import OpenAI

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    print(f"  🌐 DeepSeek API: {base_url}")
    print(f"  🤖 模型: {model}")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=6000,
        messages=[
            {
                "role": "system",
                "content": "你是精通 Ramsey Theory 的组合数学专家，擅长用中文撰写学术论文解读。",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def fallback_digest(papers: List[Paper]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:3])
        summary = " ".join(p.abstract.split())[:200] + "..."
        lines.append(
            f"### 论文 {i}: {p.title}\n"
            f"- 来源: {p.source}\n"
            f"- 作者: {authors}\n"
            f"- 链接: {p.url}\n"
            f"- 摘要: {summary}\n"
        )
    return "\n".join(lines)


def generate_ai_digest(papers: List[Paper]) -> str:
    if not papers:
        return "今日暂无符合条件的 Ramsey Theory 新论文。"

    prompt = build_prompt(papers)
    print(f"📝 Prompt 长度: {len(prompt)} 字符")

    ai_provider = os.environ.get("AI_PROVIDER", "deepseek")

    try:
        if ai_provider == "deepseek":
            print("🤖 调用 DeepSeek API...")
            return call_deepseek(prompt)
        else:
            # 保留 Claude 兼容性（如果需要）
            print("🤖 调用 Claude API...")
            # 这里可以保留 call_claude 函数，但你不需要可以删除
            return fallback_digest(papers)
    except Exception as e:
        print(f"❌ AI 生成失败: {e}")
        traceback.print_exc()
        return fallback_digest(papers)


# ============================================================
#  发送邮件
# ============================================================
def send_email(digest: str, count: int, source_stats: dict):
    SENDER = os.environ["SENDER_EMAIL"]
    SMTP_PASSWORD = os.environ["SENDER_APP_PASSWORD"]
    RECEIVER = os.environ.get("RECEIVER_EMAIL", SENDER)
    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.163.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", 465))

    # 支持多个收件人（逗号分隔）
    receivers = [r.strip() for r in RECEIVER.split(",")] if RECEIVER else [SENDER]

    try:
        import markdown as md_lib
        html_content = md_lib.markdown(digest, extensions=["extra"])
    except ImportError:
        html_content = digest.replace("\n", "<br>")

    stats_html = " | ".join(f"{src}: {cnt}" for src, cnt in source_stats.items())

    html_body = f"""
    <html><body style="font-family:Arial,sans-serif; max-width:760px;
                       margin:auto; padding:20px; color:#333; line-height:1.7;">
    <h1 style="color:#1565C0;">Ramsey Theory 每日论文解读</h1>
    <p style="color:#666;">{DATE_STR} | 共 {count} 篇 | {stats_html}</p>
    <hr>
    {html_content}
    <hr>
    <p style="color:#aaa; font-size:11px;">
    数据源: arXiv + Semantic Scholar + OpenAlex + 期刊RSS + MathOverflow<br>
    AI: DeepSeek
    </p>
    </body></html>
    """

    msg = EmailMessage()
    msg["Subject"] = f"Ramsey Theory 每日解读 {DATE_STR}（{count}篇）"
    msg["From"] = formataddr(("Ramsey Digest", SENDER))
    msg["To"] = ", ".join(receivers)
    msg.set_content(digest)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SENDER, SMTP_PASSWORD)
        server.send_message(msg)

    print(f"邮件已发送到: {', '.join(receivers)}")


# ============================================================
#  主函数
# ============================================================
def main():
    print(f"=== Ramsey Theory 全源每日论文解读 ({DATE_STR}) ===")

    # 第一步：多源抓取
    all_papers = []
    source_stats = {}

    fetchers = [
        ("arXiv", fetch_arxiv),
        ("Semantic Scholar", fetch_semantic_scholar),
        ("OpenAlex", fetch_openalex),
        ("Journal RSS", fetch_journal_rss),
        ("MathOverflow", fetch_mathoverflow),
    ]

    for name, fetcher in fetchers:
        try:
            papers = fetcher()
            source_stats[name] = len(papers)
            all_papers.extend(papers)
        except Exception as e:
            print(f"{name} 失败: {e}")
            source_stats[name] = 0

    # 第二步：去重
    unique_papers = deduplicate(all_papers)
    total = len(unique_papers)

    print(f"\n来源统计: {source_stats}")
    print(f"去重后总计: {total} 篇")

    # 第三步：AI 解读
    if unique_papers:
        digest = generate_ai_digest(unique_papers)
    else:
        digest = f"# {DATE_STR} Ramsey Theory 日报\n\n今日暂无新论文。"

    print(f"解读长度: {len(digest)} 字符")

    # 第四步：发送邮件
    try:
        send_email(digest, total, source_stats)
    except Exception as e:
        print(f"邮件发送失败: {e}")
        traceback.print_exc()

    print("=== 完成 ===")


if __name__ == "__main__":
    main()
