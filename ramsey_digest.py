import os
import re
import json
import time
import arxiv
import requests
import smtplib
import hashlib
import traceback
from email.message import EmailMessage
from email.utils import formataddr
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone


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
    def fingerprint(self) -> str:
        normalized = re.sub(r"[^a-z0-9]", "", self.title.lower())
        return hashlib.md5(normalized.encode()).hexdigest()


now_utc = datetime.now(timezone.utc)
DATE_STR = now_utc.strftime("%Y-%m-%d")

# 搜索窗口：过去7天（确保不漏论文）
SEARCH_WINDOW = timedelta(days=7)
since_utc = now_utc - SEARCH_WINDOW

# 论文日期底线：不接受超过30天的论文
MAX_AGE = timedelta(days=30)
oldest_allowed = now_utc - MAX_AGE

RAMSEY_KEYWORDS = [
    "Ramsey",
    "Gallai-Ramsey",
    "Ramsey number",
    "Ramsey multiplicity",
    "size Ramsey",
    "anti-Ramsey",
    "monochromatic subgraph",
    "Schur number",
    "Rado theorem",
    "Hales-Jewett",
    "Ramsey-type",
    "graph coloring Ramsey",
]

KEYWORD_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in RAMSEY_KEYWORDS),
    re.IGNORECASE,
)

SENT_FILE = "sent_papers.json"


def is_ramsey_related(title: str, abstract: str = "") -> bool:
    text = f"{title} {abstract}"
    return bool(KEYWORD_PATTERN.search(text))


def parse_date(date_str: str) -> Optional[datetime]:
    """尝试解析日期字符串，返回 datetime 或 None"""
    if not date_str or date_str == "unknown":
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def is_recent(date_str: str) -> bool:
    """检查日期是否在允许范围内（最近30天）"""
    dt = parse_date(date_str)
    if dt is None:
        return False
    return dt >= oldest_allowed


# ============================================================
#  去重缓存：避免重复发送
# ============================================================
def load_sent() -> Set[str]:
    try:
        with open(SENT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data)
        return set()
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_sent(fingerprints: Set[str]) -> None:
    # 这里保留最多 500 条；由于只用于去重，存 fingerprint 即可
    trimmed = sorted(fingerprints)[-500:]
    with open(SENT_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def filter_new_papers(papers: List[Paper]) -> Tuple[List[Paper], Set[str]]:
    """
    只负责筛选，不负责写入缓存。
    返回：
      - new_papers: 本次真正的新论文
      - sent: 旧缓存中的已发送 fingerprint 集合
    """
    sent = load_sent()
    new_papers = [p for p in papers if p.fingerprint not in sent]
    print(
        f"Filter: {len(papers)} total, {len(new_papers)} new, "
        f"{len(papers) - len(new_papers)} already sent"
    )
    return new_papers, sent


def update_sent_cache(sent: Set[str], papers: List[Paper]) -> None:
    """
    仅在邮件发送成功后调用。
    """
    updated = sent | {p.fingerprint for p in papers}
    save_sent(updated)
    print(f"sent_papers.json 已更新，当前缓存条目数: {min(len(updated), 500)}")


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
        ")"
    )
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=80,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers: List[Paper] = []
    try:
        for r in client.results(search):
            pub_date = r.published.replace(tzinfo=timezone.utc)
            if pub_date < since_utc:
                continue
            papers.append(
                Paper(
                    title=r.title,
                    authors=[a.name for a in r.authors],
                    abstract=r.summary.strip(),
                    url=r.entry_id,
                    source="arXiv",
                    published=r.published.strftime("%Y-%m-%d"),
                    doi=r.doi,
                )
            )
    except Exception as e:
        print(f"  error: {e}")
    print(f"  arXiv: {len(papers)} papers")
    return papers


# ============================================================
#  Source 2: Semantic Scholar
# ============================================================
def fetch_semantic_scholar() -> List[Paper]:
    print(">>> [Semantic Scholar] querying...")
    s2_api = "https://api.semanticscholar.org/graph/v1"
    papers: List[Paper] = []
    for sq in ("Ramsey theory graph", "Ramsey number combinatorics"):
        try:
            params = {
                "query": sq,
                "year": str(now_utc.year),
                "fieldsOfStudy": "Mathematics",
                "fields": "title,authors,abstract,url,externalIds,publicationDate,venue",
                "limit": 30,
            }
            resp = requests.get(f"{s2_api}/paper/search", params=params, timeout=15)
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                pub_date = item.get("publicationDate", "")
                title = item.get("title", "")
                abstract = item.get("abstract", "") or ""
                if not is_ramsey_related(title, abstract):
                    continue
                if not is_recent(pub_date):
                    print(f"    skipped (old): {pub_date} {title[:40]}")
                    continue
                papers.append(
                    Paper(
                        title=title,
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        abstract=abstract,
                        url=item.get("url", ""),
                        source="Semantic Scholar",
                        published=pub_date or "unknown",
                        doi=(item.get("externalIds", {}) or {}).get("DOI"),
                        journal=item.get("venue", ""),
                    )
                )
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
    papers: List[Paper] = []
    try:
        params = {
            "search": "Ramsey theory graph",
            "filter": f"from_publication_date:{since_utc.strftime('%Y-%m-%d')},type:article|preprint",
            "sort": "publication_date:desc",
            "per_page": 30,
            "mailto": os.environ.get("SENDER_EMAIL", "test@example.com"),
        }
        resp = requests.get("https://api.openalex.org/works", params=params, timeout=15)
        resp.raise_for_status()
        for work in resp.json().get("results", []):
            title = work.get("title", "")
            pub_date = work.get("publication_date", "")
            if not is_recent(pub_date):
                continue

            abstract_inv = work.get("abstract_inverted_index", {})
            if abstract_inv:
                max_pos = max(pos for positions in abstract_inv.values() for pos in positions)
                words = [""] * (max_pos + 1)
                for word, positions in abstract_inv.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words)
            else:
                abstract = ""

            if not is_ramsey_related(title, abstract):
                continue

            authors = [
                a.get("author", {}).get("display_name", "")
                for a in work.get("authorships", [])[:5]
            ]
            doi_url = work.get("doi", "")
            primary_loc = work.get("primary_location") or {}

            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=primary_loc.get("landing_page_url", "") or doi_url or "",
                    source="OpenAlex",
                    published=pub_date,
                    doi=doi_url.replace("https://doi.org/", "") if doi_url else None,
                    journal=(
                        primary_loc.get("source", {}).get("display_name", "")
                        if primary_loc else ""
                    ),
                )
            )
    except Exception as e:
        print(f"  OpenAlex error: {e}")
    print(f"  OpenAlex: {len(papers)} papers")
    return papers


# ============================================================
#  Source 4: Journal RSS
# ============================================================
def fetch_journal_rss() -> List[Paper]:
    print(">>> [Journal RSS] querying...")
    rss_feeds = {
        "J. Combin. Theory B": "https://rss.sciencedirect.com/publication/science/00958956",
        "Discrete Math": "https://rss.sciencedirect.com/publication/science/0012365X",
        "European J. Combin.": "https://rss.sciencedirect.com/publication/science/01956698",
        "Electron. J. Combin.": "https://www.combinatorics.org/ojs/index.php/eljc/gateway/plugin/WebFeedGatewayPlugin/atom",
    }
    papers: List[Paper] = []
    try:
        import feedparser
    except ImportError:
        print("  feedparser not installed, skipping")
        return papers

    for journal_name, feed_url in rss_feeds.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                summary = re.sub(r"<[^>]+>", "", entry.get("summary", "") or "").strip()

                if not is_ramsey_related(title, summary):
                    continue

                pub_date = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = time.strftime("%Y-%m-%d", entry.published_parsed)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    pub_date = time.strftime("%Y-%m-%d", entry.updated_parsed)

                if pub_date and not is_recent(pub_date):
                    print(f"    skipped (old): {pub_date} {title[:40]}")
                    continue
                if not pub_date:
                    print(f"    skipped (no date): {title[:40]}")
                    continue

                authors = (
                    [a.get("name", "") for a in entry.authors]
                    if hasattr(entry, "authors")
                    else ([entry.author] if hasattr(entry, "author") else [])
                )

                papers.append(
                    Paper(
                        title=title,
                        authors=authors,
                        abstract=summary[:500],
                        url=entry.get("link", ""),
                        source=f"Journal:{journal_name}",
                        published=pub_date,
                        journal=journal_name,
                    )
                )
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
    papers: List[Paper] = []
    try:
        params = {
            "order": "desc",
            "sort": "creation",
            "q": "Ramsey",
            "tagged": "combinatorics",
            "site": "mathoverflow",
            "fromdate": int(since_utc.timestamp()),
            "pagesize": 10,
        }
        resp = requests.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        for item in resp.json().get("items", []):
            title = item.get("title", "")
            body = re.sub(r"<[^>]+>", "", item.get("body", ""))[:400]
            if not is_ramsey_related(title, body):
                continue
            papers.append(
                Paper(
                    title=f"[MO] {title}",
                    authors=[item.get("owner", {}).get("display_name", "anonymous")],
                    abstract=body,
                    url=item.get("link", ""),
                    source="MathOverflow",
                    published=datetime.fromtimestamp(
                        item["creation_date"], tz=timezone.utc
                    ).strftime("%Y-%m-%d"),
                )
            )
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
        papers_text += f"""
---
论文 {i}:
标题: {p.title}
作者: {authors}
来源: {p.source}
日期: {p.published}
期刊: {p.journal or 'preprint'}
链接: {p.url}
摘要: {" ".join(p.abstract.split())}
---
"""
    return f"""你是图论与 Ramsey Theory 方向的数学研究专家，精通中文学术写作。

以下是今日（{DATE_STR}）汇总的 {len(papers)} 篇 Ramsey Theory 相关最新论文。

对每篇论文请输出：

### 📄 论文 X：[中文标题]
- **原标题**：
- **作者**：
- **来源**：
- **链接**：
- **中文摘要**（3-5句）
- **主要贡献**
- **技术方法**
- **学术脉络**
- **推荐指数**：⭐-⭐⭐⭐⭐⭐

最后给出「今日综述」：今日趋势 + 最值得关注的论文。

论文列表：
{papers_text}

请用中文和 Markdown 格式输出。"""


# ============================================================
#  AI 调用
# ============================================================
def call_deepseek(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    print(f"  API Key 前6位: {api_key[:6] if api_key else '空！'}")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未配置！")

    client = OpenAI(api_key=api_key, base_url=base_url)
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
    print(f"  Token 使用: {response.usage}")
    return response.choices[0].message.content


def fallback_digest(papers: List[Paper]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(
            f"### 论文 {i}: {p.title}\n"
            f"- 来源: {p.source}\n"
            f"- 作者: {', '.join(p.authors[:3])}\n"
            f"- 链接: {p.url}\n"
            f"- 摘要: {' '.join(p.abstract.split())[:200]}...\n"
        )
    return "\n".join(lines)


def generate_ai_digest(papers: List[Paper]) -> str:
    if not papers:
        return "今日暂无符合条件的 Ramsey Theory 新论文。"

    prompt = build_prompt(papers)
    print(f"Prompt 长度: {len(prompt)} 字符")
    print(">>> 调用 DeepSeek API...")

    try:
        result = call_deepseek(prompt)
        print(">>> DeepSeek 调用成功！")
        return result
    except Exception as e:
        print(f">>> DeepSeek 调用失败: {e}")
        traceback.print_exc()
        return fallback_digest(papers)


# ============================================================
#  发送邮件
# ============================================================
def send_email(digest: str, count: int, source_stats: dict) -> None:
    sender = os.environ["SENDER_EMAIL"]
    smtp_password = os.environ["SENDER_APP_PASSWORD"]
    receiver = os.environ.get("RECEIVER_EMAIL", sender)
    smtp_host = os.environ.get("SMTP_HOST", "smtp.163.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 465))

    receivers = [r.strip() for r in receiver.split(",")] if receiver else [sender]

    try:
        import markdown as md_lib
        html_content = md_lib.markdown(digest, extensions=["extra"])
    except ImportError:
        html_content = digest.replace("\n", "<br>")

    stats_html = " | ".join(f"{src}: {cnt}" for src, cnt in source_stats.items())
    html_body = f"""
    <html>
      <body style="font-family:Arial,sans-serif;max-width:760px;margin:auto;padding:20px;color:#333;line-height:1.7;">
        <h1 style="color:#1565C0;">Ramsey Theory 每日论文解读</h1>
        <p style="color:#666;">{DATE_STR} | 共 {count} 篇新论文 | {stats_html}</p>
        <hr>
        {html_content}
        <hr>
        <p style="color:#aaa;font-size:11px;">数据源: arXiv + Semantic Scholar + OpenAlex + 期刊RSS + MathOverflow | AI: DeepSeek</p>
      </body>
    </html>
    """

    msg = EmailMessage()
    msg["Subject"] = f"Ramsey Theory 每日解读 {DATE_STR}（{count}篇新论文）"
    msg["From"] = formataddr(("Ramsey Digest", sender))
    msg["To"] = ", ".join(receivers)
    msg.set_content(digest)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(sender, smtp_password)
        server.send_message(msg)

    print(f"邮件已发送到: {', '.join(receivers)}")


# ============================================================
#  主函数
# ============================================================
def main() -> None:
    print(f"=== Ramsey Theory 每日论文解读 ({DATE_STR}) ===")
    print(f"搜索窗口: {since_utc.strftime('%Y-%m-%d')} ~ {now_utc.strftime('%Y-%m-%d')}")
    print(f"最旧允许: {oldest_allowed.strftime('%Y-%m-%d')}")

    all_papers: List[Paper] = []
    source_stats = {}

    for name, fetcher in [
        ("arXiv", fetch_arxiv),
        ("Semantic Scholar", fetch_semantic_scholar),
        ("OpenAlex", fetch_openalex),
        ("Journal RSS", fetch_journal_rss),
        ("MathOverflow", fetch_mathoverflow),
    ]:
        try:
            papers = fetcher()
            source_stats[name] = len(papers)
            all_papers.extend(papers)
        except Exception as e:
            print(f"{name} 失败: {e}")
            source_stats[name] = 0

    unique_papers = deduplicate(all_papers)
    unique_papers, sent = filter_new_papers(unique_papers)
    total = len(unique_papers)

    print(f"\n来源统计: {source_stats}")
    print(f"最终新论文数: {total} 篇")

    digest = (
        generate_ai_digest(unique_papers)
        if unique_papers
        else f"# {DATE_STR}\n\n今日暂无新论文。"
    )
    print(f"解读长度: {len(digest)} 字符")

    try:
        send_email(digest, total, source_stats)
        update_sent_cache(sent, unique_papers)
    except Exception as e:
        print(f"邮件发送失败: {e}")
        traceback.print_exc()
        raise

    print("=== 完成 ===")


if __name__ == "__main__":
    main()
