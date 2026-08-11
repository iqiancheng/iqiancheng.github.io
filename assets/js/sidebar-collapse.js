(function () {
  var KEY = 'sidebar-collapsed';
  var root = document.documentElement;
  var btn = document.getElementById('sidebar-collapse-btn');
  if (!btn) return;

  function apply(collapsed) {
    root.setAttribute('data-sidebar-collapsed', collapsed ? 'true' : 'false');
    var icon = btn.querySelector('i');
    if (icon) {
      icon.className = collapsed ? 'fas fa-angle-right' : 'fas fa-angle-left';
    }
  }

  apply(root.getAttribute('data-sidebar-collapsed') === 'true');

  btn.addEventListener('click', function () {
    var next = root.getAttribute('data-sidebar-collapsed') !== 'true';
    apply(next);
    try {
      localStorage.setItem(KEY, next ? 'true' : 'false');
    } catch (e) {}
  });
})();
