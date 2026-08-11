(function () {
  var KEY = 'sidebar-collapsed';
  var root = document.documentElement;
  var trigger = document.getElementById('sidebar-trigger');
  if (!trigger) return;

  var mq = window.matchMedia('(min-width: 850px)');

  function isDesktop() {
    return mq.matches;
  }

  function collapsed() {
    return root.getAttribute('data-sidebar-collapsed') === 'true';
  }

  function setIcon() {
    var icon = trigger.querySelector('i');
    if (icon) {
      icon.className = 'fas fa-bars fa-fw';
    }
  }

  function apply(next) {
    root.setAttribute('data-sidebar-collapsed', next ? 'true' : 'false');
    setIcon();
    try {
      localStorage.setItem(KEY, next ? 'true' : 'false');
    } catch (e) {}
  }

  function initDesktop() {
    var saved = localStorage.getItem(KEY) === 'true';
    root.setAttribute('data-sidebar-collapsed', saved ? 'true' : 'false');
    setIcon();
  }

  var themeOnclick = trigger.onclick;

  function install() {
    themeOnclick = trigger.onclick;
    trigger.onclick = function (e) {
      if (!isDesktop()) {
        if (typeof themeOnclick === 'function') {
          return themeOnclick.call(this, e);
        }
        return true;
      }
      e && e.preventDefault();
      apply(!collapsed());
      return false;
    };
  }

  function onBreakpoint() {
    if (isDesktop()) {
      initDesktop();
    } else {
      root.removeAttribute('data-sidebar-collapsed');
      setIcon();
    }
  }

  if (isDesktop()) {
    initDesktop();
  }

  if (mq.addEventListener) {
    mq.addEventListener('change', onBreakpoint);
  } else {
    mq.addListener(onBreakpoint);
  }

  if (document.readyState === 'complete') {
    install();
  } else {
    window.addEventListener('load', install);
  }
})();
