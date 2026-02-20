window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    processHtmlClass: "arithmatex|jp-RenderedHTMLCommon|jp-RenderedMarkdown",
  },
  startup: {
    typeset: false,
  },
};

function typesetMath() {
  if (!window.MathJax || !window.MathJax.startup) {
    return;
  }

  window.MathJax.startup.promise
    .then(() => {
      window.MathJax.startup.output.clearCache();
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      return window.MathJax.typesetPromise();
    })
    .catch((error) => {
      console.error("MathJax typeset failed:", error);
    });
}

if (window.document$ && typeof window.document$.subscribe === "function") {
  window.document$.subscribe(typesetMath);
} else {
  window.addEventListener("DOMContentLoaded", typesetMath);
}
