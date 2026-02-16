/* pitcher.js - Individual pitcher detail page */
(function () {
  'use strict';

  // Pitch type display names
  var PITCH_NAMES = {
    FF: '4-Seam FB', SI: 'Sinker', FC: 'Cutter', CH: 'Changeup',
    SL: 'Slider', CU: 'Curveball', KC: 'Knuckle Curve', ST: 'Sweeper',
    SV: 'Slurve', FS: 'Splitter', KN: 'Knuckleball', FO: 'Forkball',
    SC: 'Screwball'
  };

  // Zone heatmap colors
  function getZoneColor(skill, nPitches) {
    if (nPitches < 10 || skill === null || skill === undefined) {
      return '#334155'; // border gray for no-data
    }
    // Interpolate: red (-0.10) -> white (0) -> green (+0.10)
    var clamped = Math.max(-0.10, Math.min(0.10, skill));
    var t = (clamped + 0.10) / 0.20; // 0 (red) to 1 (green)

    if (t < 0.5) {
      // Red to white
      var r2w = t / 0.5;
      var r = 239, g = Math.round(68 + (244 - 68) * r2w), b = Math.round(68 + (244 - 68) * r2w);
      return 'rgb(' + r + ',' + g + ',' + b + ')';
    } else {
      // White to green
      var w2g = (t - 0.5) / 0.5;
      var r = Math.round(244 - (244 - 74) * w2g), g = Math.round(244 - (244 - 222) * w2g), b = Math.round(244 - (244 - 128) * w2g);
      return 'rgb(' + r + ',' + g + ',' + b + ')';
    }
  }

  function getTextColor(skill) {
    if (skill === null || skill === undefined) return '#94a3b8';
    return Math.abs(skill) > 0.04 ? '#fff' : '#1e293b';
  }

  function pct(val) {
    return (val * 100).toFixed(1) + '%';
  }

  function signedSkill(val) {
    if (val === null || val === undefined) return '-';
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

  // Get pitcher ID from URL
  var params = new URLSearchParams(window.location.search);
  var pitcherId = params.get('id');

  if (!pitcherId) {
    document.getElementById('pitcher-content').innerHTML =
      '<p style="color:var(--negative);">No pitcher ID specified. <a href="index.html">Return to leaderboard</a>.</p>';
    return;
  }

  // Load all data in parallel
  Promise.all([
    fetch('data/pitchers_overall.json').then(function (r) { return r.json(); }),
    fetch('data/pitchers_by_pitch_type.json').then(function (r) { return r.json(); }),
    fetch('data/pitchers_by_zone.json').then(function (r) { return r.json(); }),
  ])
  .then(function (results) {
    var overallData = results[0];
    var pitchTypeData = results[1];
    var zoneData = results[2];

    // Find pitcher in overall data
    var pitcher = null;
    for (var i = 0; i < overallData.pitchers.length; i++) {
      if (String(overallData.pitchers[i].pitcher_id) === pitcherId) {
        pitcher = overallData.pitchers[i];
        break;
      }
    }

    if (!pitcher) {
      document.getElementById('pitcher-content').innerHTML =
        '<p style="color:var(--negative);">Pitcher not found (ID: ' + escapeHtml(pitcherId) + '). <a href="index.html">Return to leaderboard</a>.</p>';
      return;
    }

    document.title = pitcher.name + ' - Whiff Skill';

    var pitchTypes = pitchTypeData[pitcherId] || {};

    // Convert compact zone format to objects
    var zoneOrder = zoneData.zone_order || [];
    var rawZones = zoneData.pitchers ? zoneData.pitchers[pitcherId] : null;
    var zones = [];
    if (rawZones && zoneOrder.length) {
      for (var zi = 0; zi < zoneOrder.length; zi++) {
        var zArr = rawZones[zi];
        zones.push({
          zone_id: zoneOrder[zi],
          n_pitches: zArr[0],
          n_swings: zArr[1],
          whiff_rate: zArr[2],
          expected: zArr[3],
          skill: zArr[4],
        });
      }
    }

    renderPitcher(pitcher, pitchTypes, zones);
  })
  .catch(function (err) {
    document.getElementById('pitcher-content').innerHTML =
      '<p style="color:var(--negative);">Failed to load data: ' + err.message + '</p>';
  });

  function renderPitcher(pitcher, pitchTypes, zones) {
    var html = '';

    // Header
    html += '<div class="pitcher-header">';
    html += '<h2>' + escapeHtml(pitcher.name) + '</h2>';
    html += '<span class="team-badge">' + escapeHtml(pitcher.team) + '</span>';
    html += '</div>';

    // Stat cards
    html += '<div class="stat-cards">';
    html += statCard('Pitches', pitcher.n_pitches.toLocaleString());
    html += statCard('Swings', pitcher.n_swings.toLocaleString());
    html += statCard('Whiff%', pct(pitcher.actual_whiff_rate));
    html += statCard('xWhiff%', pct(pitcher.expected_whiff_rate));
    html += statCard('Skill', signedSkill(pitcher.whiff_skill), skillClass(pitcher.whiff_skill));
    html += statCard('Percentile', pitcher.percentile + 'th');
    html += '</div>';

    // Pitch type chart
    html += '<div class="chart-section">';
    html += '<h3>Whiff Skill by Pitch Type</h3>';
    html += '<div class="chart-container" style="height:' + Math.max(200, Object.keys(pitchTypes).length * 50) + 'px;">';
    html += '<canvas id="pitch-type-chart"></canvas>';
    html += '</div></div>';

    // Zone heatmap
    html += '<div class="chart-section">';
    html += '<h3>Whiff Skill by Zone</h3>';
    html += '<p style="font-size:0.8rem;color:var(--text-muted);margin-bottom:1rem;">From the batter\'s perspective. Green = more whiffs than expected. Red = fewer.</p>';
    html += renderZoneGrid(zones);
    html += '</div>';

    // Back link
    html += '<p><a href="index.html">&larr; Back to Leaderboard</a></p>';

    document.getElementById('pitcher-content').innerHTML = html;

    // Render chart after DOM update
    renderPitchTypeChart(pitchTypes);
  }

  function statCard(label, value, cls) {
    return '<div class="stat-card"><div class="label">' + label + '</div>' +
      '<div class="value ' + (cls || '') + '">' + value + '</div></div>';
  }

  function renderPitchTypeChart(pitchTypes) {
    var canvas = document.getElementById('pitch-type-chart');
    if (!canvas) return;

    var types = Object.keys(pitchTypes);
    if (types.length === 0) {
      canvas.parentElement.innerHTML = '<p style="color:var(--text-muted);">No pitch type data available.</p>';
      return;
    }

    // Sort by skill descending
    types.sort(function (a, b) { return pitchTypes[b].skill - pitchTypes[a].skill; });

    var labels = types.map(function (t) {
      return (PITCH_NAMES[t] || t) + ' (' + pitchTypes[t].n_swings + ')';
    });
    var skills = types.map(function (t) { return pitchTypes[t].skill; });
    var colors = skills.map(function (s) { return s > 0 ? '#4ade80' : '#f87171'; });

    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Whiff Skill',
          data: skills,
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
            beginAtZero: true,
            grid: { color: '#334155' },
            ticks: { color: '#94a3b8' },
            title: {
              display: true,
              text: 'Whiff Skill (actual - expected)',
              color: '#94a3b8',
            }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#e2e8f0' },
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: function (ctx) {
                var t = types[ctx.dataIndex];
                var d = pitchTypes[t];
                return [
                  'Pitches: ' + d.n_pitches + ' (' + d.n_swings + ' swings)',
                  'Whiff%: ' + (d.whiff_rate * 100).toFixed(1) + '%',
                  'xWhiff%: ' + (d.expected * 100).toFixed(1) + '%',
                ];
              }
            }
          }
        }
      }
    });
  }

  // Zone order: top-left to bottom-right, matching CSS grid
  // Grid rows (top to bottom): high, middle, low
  // Grid cols (left to right): inside, middle, away (batter's perspective)
  var ZONE_ORDER = [
    'high_inside', 'high_middle', 'high_away',
    'middle_inside', 'middle_middle', 'middle_away',
    'low_inside', 'low_middle', 'low_away',
  ];

  var ZONE_LABELS_X = ['Inside', 'Middle', 'Away'];
  var ZONE_LABELS_Y = ['High', 'Middle', 'Low'];

  function renderZoneGrid(zones) {
    // Index zones by zone_id
    var zoneMap = {};
    for (var i = 0; i < zones.length; i++) {
      zoneMap[zones[i].zone_id] = zones[i];
    }

    var html = '<div class="zone-wrapper">';

    // Y labels
    html += '<div class="zone-y-labels">';
    for (var yi = 0; yi < ZONE_LABELS_Y.length; yi++) {
      html += '<span>' + ZONE_LABELS_Y[yi] + '</span>';
    }
    html += '</div>';

    // Grid
    html += '<div style="flex:1;max-width:360px;">';
    html += '<div class="zone-grid">';
    for (var j = 0; j < ZONE_ORDER.length; j++) {
      var zid = ZONE_ORDER[j];
      var z = zoneMap[zid] || { zone_id: zid, n_pitches: 0, n_swings: 0, whiff_rate: null, expected: null, skill: null };

      var bg = getZoneColor(z.skill, z.n_pitches);
      var tc = getTextColor(z.skill);

      if (z.skill === null || z.skill === undefined || z.n_pitches < 10) {
        html += '<div class="zone-cell no-data" title="' + zid.replace('_', ' ') + ': ' + z.n_pitches + ' pitches (below min)">';
        html += '<span style="font-size:0.7rem;">N/A</span>';
        html += '<span class="zone-n">' + z.n_pitches + ' pit</span>';
        html += '</div>';
      } else {
        html += '<div class="zone-cell" style="background:' + bg + ';color:' + tc + ';" ';
        html += 'title="' + zid.replace('_', ' ') + '\nPitches: ' + z.n_pitches +
          '\nSwings: ' + z.n_swings +
          '\nWhiff%: ' + pct(z.whiff_rate) +
          '\nxWhiff%: ' + pct(z.expected) +
          '\nSkill: ' + signedSkill(z.skill) + '">';
        html += '<span class="skill-val">' + signedSkill(z.skill) + '</span>';
        html += '<span class="zone-n">' + z.n_swings + ' sw</span>';
        html += '</div>';
      }
    }
    html += '</div>';

    // X labels
    html += '<div class="zone-x-labels">';
    for (var xi = 0; xi < ZONE_LABELS_X.length; xi++) {
      html += '<span>' + ZONE_LABELS_X[xi] + '</span>';
    }
    html += '</div>';

    html += '</div>'; // flex container
    html += '</div>'; // zone-wrapper

    return html;
  }
})();
