/* playoffs.js - Playoff leaderboard and risers/fallers chart */
(function () {
  'use strict';

  function pct(val) {
    return (val * 100).toFixed(1) + '%';
  }

  function signedSkill(val) {
    return (val > 0 ? '+' : '') + val.toFixed(4);
  }

  function skillClass(val) {
    if (val > 0.005) return 'skill-positive';
    if (val < -0.005) return 'skill-negative';
    return 'skill-neutral';
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function deltaColor(val) {
    if (val > 0.05) return '#22c55e';   // big riser
    if (val > 0.02) return '#86efac';   // riser
    if (val > -0.02) return '#9ca3af';  // neutral
    if (val > -0.05) return '#fca5a5';  // faller
    return '#ef4444';                    // big faller
  }

  // Load playoff data
  Promise.all([
    fetch('data/pitchers_playoffs.json').then(function (r) { return r.json(); }),
    fetch('data/playoff_comparison.json').then(function (r) { return r.json(); }),
  ])
  .then(function (results) {
    var poData = results[0];
    var compData = results[1];

    renderPoLeaderboard(poData.pitchers);
    renderComparisonChart(compData.pitchers);
  })
  .catch(function (err) {
    document.getElementById('po-table-body').innerHTML =
      '<tr><td colspan="9" style="text-align:center;padding:2rem;color:var(--negative);">Failed to load data: ' + err.message + '</td></tr>';
  });

  function renderPoLeaderboard(pitchers) {
    var tbody = document.getElementById('po-table-body');

    if (pitchers.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:2rem;color:var(--text-muted);">No playoff data available.</td></tr>';
      return;
    }

    var html = '';
    for (var i = 0; i < pitchers.length; i++) {
      var p = pitchers[i];
      var sc = skillClass(p.whiff_skill);
      var rsWarning = !p.rs_qualified ? ' title="Below 250 RS pitch threshold"' : '';
      var nameDisplay = escapeHtml(p.name) + (!p.rs_qualified ? ' <span style="color:#facc15;">&#9888;</span>' : '');

      html += '<tr onclick="window.location=\'pitcher.html?id=' + p.pitcher_id + '\'"' + rsWarning + '>' +
        '<td class="num">' + (i + 1) + '</td>' +
        '<td>' + nameDisplay + '</td>' +
        '<td>' + escapeHtml(p.team) + '</td>' +
        '<td class="num">' + p.n_pitches + '</td>' +
        '<td class="num">' + p.n_swings + '</td>' +
        '<td class="num">' + pct(p.actual_whiff_rate) + '</td>' +
        '<td class="num">' + pct(p.expected_whiff_rate) + '</td>' +
        '<td class="num ' + sc + '">' + signedSkill(p.whiff_skill) + '</td>' +
        '<td class="num"><span class="confidence-badge ' + p.confidence + '">' + p.confidence + '</span></td>' +
        '</tr>';
    }
    tbody.innerHTML = html;
  }

  function renderComparisonChart(pitchers) {
    var canvas = document.getElementById('comparison-chart');
    var container = document.getElementById('comparison-chart-container');

    if (pitchers.length === 0) {
      container.innerHTML = '<p style="color:var(--text-muted);">No pitchers qualify for RS/PO comparison (need 250+ RS and 50+ PO pitches).</p>';
      return;
    }

    // Set dynamic height based on number of pitchers
    container.style.height = Math.max(300, pitchers.length * 40 + 60) + 'px';

    var labels = pitchers.map(function (p) {
      var conf = p.confidence === 'low' ? ' \u26A0' : '';
      return p.name + conf;
    });
    var deltas = pitchers.map(function (p) { return p.skill_delta; });
    var colors = deltas.map(function (d) { return deltaColor(d); });

    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Playoff Skill Change',
          data: deltas,
          backgroundColor: colors,
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: '#334155' },
            ticks: { color: '#94a3b8' },
            title: {
              display: true,
              text: 'Skill Delta (PO - RS)',
              color: '#94a3b8',
            }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#e2e8f0', font: { size: 11 } },
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: function (ctx) {
                var p = pitchers[ctx.dataIndex];
                return [
                  'RS: ' + p.rs_pitches + ' pitches (skill: ' + signedSkill(p.rs_skill) + ')',
                  'PO: ' + p.po_pitches + ' pitches (skill: ' + signedSkill(p.po_skill) + ')',
                  'Category: ' + p.category,
                  'Confidence: ' + p.confidence,
                ];
              }
            }
          }
        }
      }
    });
  }
})();
