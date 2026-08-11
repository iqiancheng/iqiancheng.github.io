(function () {
  var input = document.getElementById('search-input');
  if (!input) return;

  var trigger = document.getElementById('search-trigger');

  function isTypingTarget(el) {
    if (!el) return false;
    var tag = el.tagName;
    return (
      tag === 'INPUT' ||
      tag === 'TEXTAREA' ||
      tag === 'SELECT' ||
      el.isContentEditable
    );
  }

  function openSearch() {
    // On <lg the inline #search is display:none — click the trigger so the
    // theme reveals the search overlay (#search -> d-flex) before focusing.
    if (trigger && getComputedStyle(input.parentElement).display === 'none') {
      trigger.click();
    }
    input.focus();
    if (input.select) input.select();
  }

  document.addEventListener('keydown', function (e) {
    var modK = (e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K');
    if (modK) {
      e.preventDefault();
      openSearch();
      return;
    }

    // Plain '/' (no modifiers) — but never when already typing somewhere.
    if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey) {
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      openSearch();
    }
  });
})();
