(function () {
  var KEY = 'sidebar-collapsed';
  var body = document.body;
  var btn = document.getElementById('sidebar-collapse-btn');
  if (!btn) return;

  function apply(collapsed) {
    body.setAttribute('data-sidebar-collapsed', collapsed ? 'true' : 'false');
    var icon = btn.querySelector('i');
    if (icon) {
      icon.className = collapsed ? 'fas fa-angle-right' : 'fas fa-angle-left';
    }
  }

  apply(localStorage.getItem(KEY) === 'true');

  btn.addEventListener('click', function () {
    var next = body.getAttribute('data-sidebar-collapsed') !== 'true';
    apply(next);
    try {
      localStorage.setItem(KEY, next ? 'true' : 'false');
    } catch (e) {}
  });
})();
