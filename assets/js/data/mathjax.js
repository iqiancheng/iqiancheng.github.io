---
layout: compress
# MathJax 3 config — site override for robust inline math next to markdown.
# WARNING: Don't use '//' to comment out code; use Liquid comments.
---

{%- comment -%}
  Goals:
  1) Prefer \(...\) / \[...\] (what kramdown/Chirpy often emit from $$...$$).
  2) Keep $$...$$ as display; avoid bare $...$ so markdown/currency/`code` collide less.
  3) Skip code/pre/rouge so backtick runs never get half-eaten by the math scanner.
  4) processEscapes: allow \* \_ \$ inside TeX without markdown stealing them.
  See: https://docs.mathjax.org/en/latest/options/input/tex.html
{%- endcomment -%}

MathJax = {
  tex: {
    inlineMath: [
      ['\\(', '\\)']
    ],
    displayMath: [
      ['$$', '$$'],
      ['\\[', '\\]']
    ],
    processEscapes: true,
    processEnvironments: true,
    packages: { '[+]': ['ams', 'noerrors', 'noundefined'] },
    tags: 'ams'
  },
  options: {
    skipHtmlTags: [
      'script', 'noscript', 'style', 'textarea', 'pre', 'code',
      'annotation', 'annotation-xml', 'kbd', 'samp'
    ],
    // Empty processHtmlClass ⇒ process whole document except ignored classes/tags.
    // Critical: do NOT process <code>/<pre> (backtick runs stay literal).
    ignoreHtmlClass: 'tex2jax_ignore|rouge|highlight|language-plaintext|language-text|no-math',
    processHtmlClass: ''
  },
  chtml: {
    scale: 1,
    displayAlign: 'center'
  },
  startup: {
    typeset: true
  }
};
