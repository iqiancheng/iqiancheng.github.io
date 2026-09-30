---
layout: compress
# MathJax 3 config — site override for robust inline math next to markdown.
# WARNING: Don't use '//' to comment out code; use Liquid comments.
---

{%- comment -%}
  Goals:
  1) Accept $...$ and \(...\) for inline math (posts write inline math as $...$).
  2) Keep $$...$$ / \[...\] as display math.
  3) Skip code/pre/rouge so backtick runs never get half-eaten by the math scanner.
  4) processEscapes: allow \* \_ \$ inside TeX without markdown stealing them.
  5) Literal currency (e.g. $10M) must be escaped as \$ in the markdown source,
     otherwise a pair of $ on one line is parsed as inline math.
  See: https://docs.mathjax.org/en/latest/options/input/tex.html
{%- endcomment -%}

MathJax = {
  tex: {
    inlineMath: [
      ['$', '$'],
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
