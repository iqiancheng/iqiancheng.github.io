# AGENTS.md — iqiancheng.github.io

Guidance for coding agents working on this Chirpy / Jekyll site.

## Stack

- Theme: `jekyll-theme-chirpy` (~7.3)
- Posts: `_posts/YYYY-MM-DD-slug.md`
- Local overrides live in `_includes/`, `_data/origin/`, `assets/` (they shadow the gem)
- Dev server is typically `http://127.0.0.1:4001/`

---

## Math formulas (critical)

Site config: `assets/js/data/mathjax.js`  
Lazy loader: `_includes/js-selector.html` (only when front matter has `math: true`)

### Root cause of broken inline math

MathJax **skips** `<code>`, `<pre>`, and Rouge highlight wrappers.

If you wrap a formula in backticks:

```markdown
`θ_i = base^(−2i/d)`
```

Kramdown emits `<code>...</code>` → MathJax never typesets it. This is the usual failure mode for “公式不渲染”.

Other failure modes:

| Pattern | What goes wrong |
|---------|-----------------|
| `**… \`math\` …**` | Formula trapped inside bold+code |
| Bare `$...$` | Disabled on purpose (collides with markdown / currency) |
| `\$$...$$` mis-escaped in lists | Can garble delimiters into `$$…\(…\\)…$$` |
| `math: true` missing | MathJax script never loaded |
| Formula only in backticks on a `math: true` page | Still skipped (code tag) |

### Writing conventions (do this)

**Enable MathJax on the post:**

```yaml
---
layout: post
title: "..."
date: YYYY-MM-DD HH:MM:SS +0800
math: true
---
```

**Inline math** — Chirpy style `$$...$$` (no blank lines before/after). Prefer **outside** bold, with a clear break:

```markdown
**2. 频率公式** $$\theta_i = \mathrm{base}^{(-2i/d)}$$。第 $$i$$ 个坐标对…
```

Good:

```markdown
因为 $$H(P)$$ 对模型参数是常数
旋转矩阵 $$R(m)$$，相对位置 $$n-m$$
范围 $$[-1,1]$$
```

**Display / block math** — blank lines around the fence:

```markdown
$$
D_{KL}(P \parallel Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$
```

**Lists** — use normal `$$...$$` (this site’s kramdown path already turns them into `\(...\)`). Do **not** invent extra `\$$` escapes unless you verify HTML after build.

**Mermaid** (separate from math):

```yaml
mermaid: true   # only if the body contains ```mermaid fences
```

### Fix methods (when formulas break)

1. **Confirm front matter**  
   - `math: true` present  
   - `layout: post` (not `layout: default`)

2. **Confirm source is not code**  
   - Search the post for `` `...` `` around Greek letters / `^` / `_`  
   - Convert real formulas to `$$...$$`  
   - Leave true code identifiers in backticks (`input_ids`, `base_url`, …)

3. **Separate markdown emphasis from math**  
   - Bad: `**2. 频率公式 \`θ_i = ...\`。**`  
   - Good: `**2. 频率公式** $$\theta_i = \mathrm{base}^{(-2i/d)}$$。`

4. **Use TeX, not Unicode-as-code**  
   - `` `θ_i` `` → `$$\theta_i$$`  
   - `` `base^(−2i/d)` `` → `$$\mathrm{base}^{(-2i/d)}$$`  
   - Prefer `\theta`, `\mathrm{base}`, `^{...}` over raw `θ` / `base^` inside math

5. **Verify built HTML** (not only the Markdown preview):

```bash
# Should see \(...\) or \[...\], NOT <code>…</code>
rg -n '\\\\(|theta|MathJax|language-plaintext' _site/posts/<slug>/index.html | head
```

6. **Hard-refresh the browser** after rebuild (lazy MathJax loads after paint).

7. **MathJax config knobs** (`assets/js/data/mathjax.js`):

   - Inline: only `\(...\)` (emitted from `$$...$$` by the pipeline)
   - Display: `$$...$$` and `\[...\]`
   - `processEscapes: true` so `\*`, `\_`, `\$` survive
   - `skipHtmlTags` includes `code` / `pre` — do not remove that just to “fix” backticks; fix the Markdown instead

### Quick conversion checklist

```text
[ ] math: true in front matter
[ ] No formula inside single backticks
[ ] **bold** closed before $$math$$ starts (or bold only the label)
[ ] Block equations have blank lines around $$
[ ] Build _site/posts/<slug>/ and confirm \( / \[ in HTML
[ ] Hard refresh; wait for lazy MathJax if formulas appear late
```

---

## Front matter conventions

```yaml
---
layout: post                 # required for post chrome / title <h1>
title: "..."                 # quote if it contains ':'
date: 2026-08-10 15:30:00 +0800   # always include +0800
tags: [llm, training]        # English kebab-case only; see tag set below
math: true                   # optional
mermaid: true                # only if ```mermaid present
---
```

### Dates

- Always `YYYY-MM-DD HH:MM:SS +0800`
- Filename date prefix should match the calendar day when possible
- UTC `...Z` and date-only forms were normalized away once; don’t reintroduce them

### Tags

- ~50 English kebab-case tags only (collapsed from a long free-form set)
- Prefer existing tags over inventing near-duplicates (`flash-attention` → `attention`, etc.)
- Full inventory lives on `/tags/` after build

---

## Performance notes (do not regress)

Post-page critical path was profiled; large wins depend on local overrides:

| File | Role |
|------|------|
| `_includes/js-selector.html` | Lazy Mermaid / MathJax; no search lib on every post |
| `_includes/search-loader.html` | Load Simple-Jekyll-Search + index on demand |
| `_includes/head.html` | `defer` theme.js; GLightbox CSS only if `<img>` |
| `_data/origin/cors.yml` | System fonts via `/assets/css/site-fonts.css` |
| `_plugins/posts-lastmod-hook.rb` | **One** `git log` for lastmod, not per-post spawns |
| `assets/js/data/search.json` | Proper `jsonify`; longer `body` field for recall |

Do **not** put `mermaid.min.js` (~2.5MB) or MathJax back into the jsDelivr combine URL for every post.

Search index must use Liquid `jsonify` (never `"{{ description }}"`) or LaTeX backslashes break JSON.

---

## Build / verify

```bash
# From repo root; prefer the project’s usual jekyll serve on :4001
bundle exec jekyll build
# or rely on running `jekyll serve` auto-regen

# Math post check
rg -n 'language-plaintext.*theta|\\\\(theta' _site/posts/multimodal-rope-decoupling/index.html

# Search index validity
python3 -c "import json; json.load(open('_site/assets/js/data/search.json')); print('ok')"
```

---

## Content sources

Posts were imported from notebook / personal blogs; older issue-style posts may still use numeric slugs (`2023-05-12-92.md`). Prefer descriptive slugs for new posts.
