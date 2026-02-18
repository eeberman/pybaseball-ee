/* metadata-banner.js - Shared metadata banner renderer */
(function () {
  'use strict';

  function normalizeSeasons(seasons) {
    if (!Array.isArray(seasons) || seasons.length === 0) return [];

    return seasons
      .map(function (s) { return parseInt(s, 10); })
      .filter(function (s) { return !isNaN(s); })
      .sort(function (a, b) { return a - b; });
  }

  function formatSeasonRange(seasons) {
    var normalized = normalizeSeasons(seasons);
    if (normalized.length === 0) return 'N/A';
    if (normalized.length === 1) return String(normalized[0]);
    return normalized[0] + '-' + normalized[normalized.length - 1];
  }

  function formatLastUpdated(value) {
    if (!value) return 'Unknown';

    var dt = new Date(value);
    if (isNaN(dt.getTime())) return value;

    return dt.toLocaleString([], {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  function defaultHeading(seasonRange, pageType) {
    if (pageType === 'playoffs') {
      return 'Data shown: 2024 Postseason (Regular Season baseline: ' + seasonRange + ')';
    }
    if (pageType === 'pitcher') {
      return 'Data shown: ' + seasonRange + ' Regular Season Pitcher Detail';
    }
    return 'Data shown: ' + seasonRange + ' Regular Season';
  }

  function defaultScopeText(seasonRange, pageType) {
    if (pageType === 'playoffs') {
      return seasonRange + ' regular-season baseline + 2024 postseason';
    }
    return seasonRange + ' regular season';
  }

  function renderMetadataBanner(options) {
    var opts = options || {};
    var banner = document.getElementById(opts.bannerId || 'metadata-banner');
    var heading = document.getElementById(opts.headingId || 'data-scope-heading');

    fetch(opts.metadataPath || 'data/metadata.json')
      .then(function (r) { return r.json(); })
      .then(function (metadata) {
        var seasonRange = formatSeasonRange(metadata.seasons_included);
        var scopeText = defaultScopeText(seasonRange, opts.pageType);

        if (banner) {
          banner.innerHTML =
            '<strong>Data scope:</strong> ' + scopeText +
            ' <span aria-hidden="true">&bull;</span> ' +
            '<strong>Last updated:</strong> ' + formatLastUpdated(metadata.last_updated);
        }

        if (heading) {
          heading.textContent = defaultHeading(seasonRange, opts.pageType);
        }
      })
      .catch(function () {
        if (banner) banner.textContent = 'Data scope metadata unavailable.';
      });
  }

  window.renderMetadataBanner = renderMetadataBanner;
})();
