/* =====================================================================
   MCEYS CUT STUDIO — UI enhancement layer
   - Does NOT modify any page function, request, or state variable.
   - Stepper: only READS the width of the existing progress bar element.
   - Drag & drop: puts the dropped file into the existing <input type=file>
     and fires its normal "change" event, so the page's own onchange
     handler runs exactly as if the user had picked the file.
   ===================================================================== */
(function () {
    'use strict';

    var STEP_LABELS = {
        server: {
            tr: ['Yükleniyor', 'Analiz', 'Kesiliyor', 'Hazır'],
            en: ['Uploading', 'Analyzing', 'Cutting', 'Ready'],
            de: ['Hochladen', 'Analyse', 'Schnitt', 'Fertig'],
            es: ['Subiendo', 'Analizando', 'Cortando', 'Listo'],
            ru: ['Загрузка', 'Анализ', 'Нарезка', 'Готово']
        },
        local: {
            tr: ['Motor', 'Belleğe Alma', 'Kesiliyor', 'Hazır'],
            en: ['Engine', 'Loading', 'Cutting', 'Ready'],
            de: ['Engine', 'Laden', 'Schnitt', 'Fertig'],
            es: ['Motor', 'Cargando', 'Cortando', 'Listo'],
            ru: ['Движок', 'Загрузка', 'Нарезка', 'Готово']
        }
    };
    // Progress % at which each step becomes the current one
    var THRESHOLDS = [1, 50, 80, 100];

    function currentLang() {
        try { return localStorage.getItem('mceys_lang') || 'tr'; } catch (e) { return 'tr'; }
    }

    function renderStepLabels() {
        var lang = currentLang();
        document.querySelectorAll('.stepper').forEach(function (stepper) {
            var set = STEP_LABELS[stepper.getAttribute('data-steps')] || STEP_LABELS.server;
            var labels = set[lang] || set.tr;
            stepper.querySelectorAll('[data-step-label]').forEach(function (el) {
                var i = parseInt(el.getAttribute('data-step-label'), 10);
                if (labels[i]) el.textContent = labels[i];
            });
        });
    }

    function initSteppers() {
        document.querySelectorAll('.stepper[data-progress-source]').forEach(function (stepper) {
            var fill = document.getElementById(stepper.getAttribute('data-progress-source'));
            if (!fill) return;
            var steps = stepper.querySelectorAll('.step');

            function sync() {
                var pct = parseFloat(fill.style.width) || 0;
                var complete = pct >= 100;
                var current = -1;
                for (var i = 0; i < THRESHOLDS.length; i++) {
                    if (pct >= THRESHOLDS[i]) current = i;
                }
                steps.forEach(function (s, i) {
                    var done = complete || i < current;
                    s.classList.toggle('done', done);
                    s.classList.toggle('active', !done && i === current);
                });
                fill.classList.toggle('is-complete', complete);
            }

            new MutationObserver(sync).observe(fill, { attributes: true, attributeFilter: ['style'] });
            sync();
        });
    }

    function acceptsFile(input, file) {
        var accept = (input.getAttribute('accept') || '').toLowerCase();
        if (!accept) return true;
        var type = (file.type || '').toLowerCase();
        if (!type) return true; // some OSes leave .mkv etc. without a MIME type
        var major = type.split('/')[0];
        return accept.split(',').some(function (tok) {
            tok = tok.trim();
            return tok === type || tok.split('/')[0] === major;
        });
    }

    function initDropZones() {
        // Only <label for="..."> upload zones; valorant.html's #drop-zone has its own handlers.
        document.querySelectorAll('label.custom-file-upload[for]').forEach(function (zone) {
            var input = document.getElementById(zone.htmlFor);
            if (!input || input.type !== 'file') return;

            ['dragenter', 'dragover'].forEach(function (ev) {
                zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add('dragover'); });
            });
            ['dragleave', 'dragend'].forEach(function (ev) {
                zone.addEventListener(ev, function () { zone.classList.remove('dragover'); });
            });
            zone.addEventListener('drop', function (e) {
                e.preventDefault();
                zone.classList.remove('dragover');
                var files = e.dataTransfer && e.dataTransfer.files;
                if (!files || !files.length || !acceptsFile(input, files[0])) return;
                try {
                    var dt = new DataTransfer();
                    dt.items.add(files[0]);
                    input.files = dt.files;
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                } catch (err) {
                    console.warn('Drop not supported in this browser:', err);
                }
            });
        });
    }

    function init() {
        renderStepLabels();
        initSteppers();
        initDropZones();
        // Re-translate stepper labels after the page's own language switch runs
        document.addEventListener('click', function (e) {
            if (e.target.closest && e.target.closest('.lang-btn')) setTimeout(renderStepLabels, 0);
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
