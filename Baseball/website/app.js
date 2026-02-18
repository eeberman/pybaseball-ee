/* app.js - Leaderboard table logic */
(function () {
  'use strict';

  let allPitchers = [];
  let sortCol = 'whiff_skill';
  let sortAsc = false;

  if (window.renderMetadataBanner) {
    window.renderMetadataBanner({ pageType: 'leaderboard' });
  }

  // --- Helpers ---
  function pct(val) {
    return (val * 100).toFixed(1) + '%';
  }

  function skillClass(val) {
    if (val > 0.005) return 'skill-positive';
    if (val < -0.005) return 'skill-negative';
    return 'skill-neutral';
  }

  function pctBarColor(p) {
    if (p >= 75) return 'var(--positive)';
    if (p >= 50) return 'var(--accent)';
    if (p >= 25) return '#facc15';
    return 'var(--negative)';
  }

  // --- Render ---
  function renderTable(pitchers) {
    var tbody = document.getElementById('table-body');
    if (pitchers.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:2rem;color:var(--text-muted);">No pitchers match your filters.</td></tr>';
      document.getElementById('result-count').textContent = '0 pitchers';
      return;
    }

    document.getElementById('result-count').textContent = pitchers.length + ' pitchers';

    var html = '';
    for (var i = 0; i < pitchers.length; i++) {
      var p = pitchers[i];
      var sc = skillClass(p.whiff_skill);
      html += '<tr onclick="window.location=\'pitcher.html?id=' + p.pitcher_id + '\'">' +
        '<td class="num">' + (i + 1) + '</td>' +
        '<td>' + escapeHtml(p.name) + '</td>' +
        '<td>' + escapeHtml(p.team) + '</td>' +
        '<td class="num">' + p.n_pitches.toLocaleString() + '</td>' +
        '<td class="num">' + p.n_swings.toLocaleString() + '</td>' +
        '<td class="num">' + pct(p.actual_whiff_rate) + '</td>' +
        '<td class="num">' + pct(p.expected_whiff_rate) + '</td>' +
        '<td class="num ' + sc + '">' + (p.whiff_skill > 0 ? '+' : '') + p.whiff_skill.toFixed(4) + '</td>' +
        '<td class="num"><span class="pct-bar"><span class="pct-bar-fill" style="width:' + p.percentile + '%;background:' + pctBarColor(p.percentile) + '"></span></span>' + p.percentile + '</td>' +
        '</tr>';
    }
    tbody.innerHTML = html;
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // --- Sort ---
  function sortPitchers(pitchers, col, asc) {
    return pitchers.slice().sort(function (a, b) {
      var va = a[col], vb = b[col];
      if (typeof va === 'string') {
        va = va.toLowerCase();
        vb = vb.toLowerCase();
      }
      if (va < vb) return asc ? -1 : 1;
      if (va > vb) return asc ? 1 : -1;
      // Secondary sort by n_pitches desc
      return b.n_pitches - a.n_pitches;
    });
  }

  function updateSortIndicators() {
    var ths = document.querySelectorAll('thead th');
    for (var i = 0; i < ths.length; i++) {
      var th = ths[i];
      var ind = th.querySelector('.sort-indicator');
      if (!ind) continue;
      if (th.dataset.col === sortCol) {
        th.classList.add('sorted');
        ind.textContent = sortAsc ? ' \u25B2' : ' \u25BC';
      } else {
        th.classList.remove('sorted');
        ind.textContent = '';
      }
    }
  }

  // --- Filter ---
  function getFiltered() {
    var search = document.getElementById('search').value.toLowerCase();
    var teamFilter = document.getElementById('team-filter').value;
    var minPitches = parseInt(document.getElementById('min-pitches').value) || 0;

    var filtered = allPitchers.filter(function (p) {
      if (search && p.name.toLowerCase().indexOf(search) === -1) return false;
      if (teamFilter && p.team !== teamFilter) return false;
      if (p.n_pitches < minPitches) return false;
      return true;
    });

    return sortPitchers(filtered, sortCol, sortAsc);
  }

  function refresh() {
    renderTable(getFiltered());
  }

  // --- Init ---
  function populateTeams(pitchers) {
    var teams = {};
    for (var i = 0; i < pitchers.length; i++) {
      teams[pitchers[i].team] = true;
    }
    var sorted = Object.keys(teams).sort();
    var sel = document.getElementById('team-filter');
    for (var j = 0; j < sorted.length; j++) {
      var opt = document.createElement('option');
      opt.value = sorted[j];
      opt.textContent = sorted[j];
      sel.appendChild(opt);
    }
  }

  fetch('data/pitchers_overall.json')
    .then(function (r) { return r.json(); })
    .then(function (data) {
      allPitchers = data.pitchers;
      populateTeams(allPitchers);

      // Default sort: whiff_skill descending
      allPitchers = sortPitchers(allPitchers, sortCol, sortAsc);
      updateSortIndicators();
      refresh();

      // Event listeners
      document.getElementById('search').addEventListener('input', refresh);
      document.getElementById('team-filter').addEventListener('change', refresh);
      document.getElementById('min-pitches').addEventListener('input', refresh);

      // Column sort
      var ths = document.querySelectorAll('thead th[data-col]');
      for (var i = 0; i < ths.length; i++) {
        ths[i].addEventListener('click', function () {
          var col = this.dataset.col;
          if (col === sortCol) {
            sortAsc = !sortAsc;
          } else {
            sortCol = col;
            // Default descending for numeric, ascending for name/team
            sortAsc = (col === 'name' || col === 'team');
          }
          updateSortIndicators();
          refresh();
        });
      }
    })
    .catch(function (err) {
      document.getElementById('table-body').innerHTML =
        '<tr><td colspan="9" style="text-align:center;padding:2rem;color:var(--negative);">Failed to load data: ' + err.message + '</td></tr>';
    });
})();
