(() => {
  const form = document.getElementById('demo-form');
  const fileInput = document.getElementById('dog_image');
  const dropzone = document.getElementById('dropzone');
  const breedInput = document.getElementById('breed');
  const breedNames = new Set([...document.querySelectorAll('#breed-options option')].map(o => o.value.toLowerCase()));
  const MEASURES = ['weight_lbs', 'neck', 'chest', 'belly', 'back'];
  const YES_NO = ['pulling', 'working', 'sports', 'growing', 'heat_sensitive', 'hiking', 'hunting', 'water', 'cleaning_experience'];
  const CHOICES = ['aggression', 'training', 'activity', 'climate', 'budget'];
  const edited = new Set();   // measurement fields the user typed themselves
  let traits = null;

  const $ = id => document.getElementById(id);
  const show = (el, on = true) => el.classList.toggle('hidden', !on);
  const icons = () => window.lucide && lucide.createIcons();
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  async function api(url, options) {
    const res = await fetch(url, options);
    let body;
    try { body = await res.json(); } catch { body = {}; }
    if (!res.ok || body.status !== 'success') throw new Error(body.message || `Request failed (${res.status})`);
    return body;
  }

  // ---------- measurements ----------
  function setBadge(field, kind) {
    const badge = document.querySelector(`[data-badge-for="${field}"]`);
    if (!badge) return;
    badge.textContent = kind === 'user' ? 'Your measurement' : 'Estimated';
    badge.className = `badge ${kind === 'user' ? 'badge-user' : 'badge-est'}`;
  }

  function applyEstimate(estimate) {
    const values = { weight_lbs: estimate.weight_lbs, ...estimate.measurements };
    const ranges = { weight_lbs: estimate.weight_range, ...(estimate.measurement_ranges || {}) };
    for (const field of MEASURES) {
      if (edited.has(field) || values[field] == null) continue;
      const input = $(field);
      input.value = values[field];
      input.title = ranges[field] ? `Typical range: ${ranges[field][0]}–${ranges[field][1]}` : '';
      setBadge(field, 'estimate');
    }
  }

  function applyTraits(newTraits, breed) {
    traits = newTraits;
    const hint = $('heat-hint');
    if (!traits || !(traits.brachycephalic || traits.coat === 'thick')) { show(hint, false); return; }
    const why = traits.brachycephalic ? 'flat-faced' : 'thick-coated';
    hint.textContent = `Suggested "Yes": ${breed}s are ${why} and can overheat easily.`;
    show(hint);
    if (!form.querySelector('input[name="heat_sensitive"]:checked')) {
      form.querySelector('input[name="heat_sensitive"][value="yes"]').checked = true;
    }
  }

  async function refreshFromBreed() {
    const breed = breedInput.value.trim();
    if (!breedNames.has(breed.toLowerCase())) return;
    const params = new URLSearchParams({ breed });
    if (edited.has('weight_lbs') && $('weight_lbs').value) params.set('weight', $('weight_lbs').value);
    try {
      const data = await api(`/api/estimate?${params}`);
      applyEstimate(data.estimate);
      applyTraits(data.traits, breed);
    } catch { /* keep current values */ }
  }

  let weightTimer;
  for (const field of MEASURES) {
    $(field).addEventListener('input', () => {
      if ($(field).value === '') { edited.delete(field); return; }
      edited.add(field);
      setBadge(field, 'user');
      if (field === 'weight_lbs') {
        clearTimeout(weightTimer);
        weightTimer = setTimeout(async () => {
          const weight = parseFloat($('weight_lbs').value);
          if (!(weight >= 1 && weight <= 250)) return;
          const params = new URLSearchParams({ weight, breed: breedInput.value.trim() });
          try { applyEstimate((await api(`/api/estimate?${params}`)).estimate); } catch { /* ignore */ }
        }, 400);
      }
    });
  }
  breedInput.addEventListener('change', refreshFromBreed);

  // ---------- photo analysis ----------
  function renderBreeds(breeds, isDog) {
    const list = $('breed-list');
    list.innerHTML = breeds.map((b, i) => `
      <li>
        <button type="button" class="breed-choice ${i === 0 && isDog ? 'is-active' : ''}" data-breed="${esc(b.name)}">
          <span class="flex justify-between text-sm font-semibold text-gray-800">
            <span>${esc(b.name)}</span><span>${Math.round(b.confidence * 100)}%</span>
          </span>
          <span class="block h-2 bg-green-100 rounded-full mt-1 overflow-hidden">
            <span class="block h-2 bg-green-600 rounded-full" style="width:${Math.max(2, Math.round(b.confidence * 100))}%"></span>
          </span>
        </button>
      </li>`).join('');
    list.querySelectorAll('.breed-choice').forEach(btn => btn.addEventListener('click', () => {
      list.querySelectorAll('.breed-choice').forEach(b => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      breedInput.value = btn.dataset.breed;
      refreshFromBreed();
    }));
  }

  async function analyze(file) {
    if (!file) return;
    show($('analysis'));
    $('image-preview').src = URL.createObjectURL(file);
    show($('analysis-loading')); show($('analysis-error'), false); show($('analysis-result'), false);

    const body = new FormData();
    body.append('image', file);
    try {
      const data = await api('/api/analyze', { method: 'POST', body });
      renderBreeds(data.breeds, data.is_dog);
      show($('analysis-result'));
      if (!data.is_dog) {
        $('analysis-error').textContent = "Hmm, we couldn't clearly spot a dog in this photo. Try a clearer photo, or pick the breed and enter measurements yourself below.";
        show($('analysis-error'));
        return;
      }
      breedInput.value = data.breeds[0].name;
      applyEstimate(data.estimate);
      applyTraits(data.traits, data.breeds[0].name);
    } catch (err) {
      $('analysis-error').textContent = err.message;
      show($('analysis-error'));
    } finally {
      show($('analysis-loading'), false);
    }
  }

  fileInput.addEventListener('change', () => analyze(fileInput.files[0]));
  ['dragenter', 'dragover'].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('is-drag'); }));
  ['dragleave', 'drop'].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('is-drag'); }));
  dropzone.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    analyze(file);
  });

  // ---------- submit ----------
  function collect() {
    const data = { breed: breedInput.value.trim() };
    const missing = [];   // elements to highlight
    for (const field of MEASURES) {
      const value = $(field).value;
      if (value !== '') data[field] = parseFloat(value);
      else if ($(field).required) missing.push($(field));
    }
    for (const name of [...YES_NO, ...CHOICES]) {
      const checked = form.querySelector(`input[name="${name}"]:checked`);
      if (checked) data[name] = checked.value;
      else missing.push(form.querySelector(`input[name="${name}"]`).closest('[role="radiogroup"]'));
    }
    return { data, missing };
  }

  const CATEGORY_META = {
    harness: { title: 'Harness', emoji: '🎽' },
    collar: { title: 'Collar', emoji: '🦮' },
    leash: { title: 'Leash', emoji: '🪢' },
    extra: { title: 'Extra', emoji: '➕' },
  };

  const linkButtons = links => links.map(l => `
    <a href="${esc(l.url)}" target="_blank" rel="noopener noreferrer" class="buy-link">
      ${esc(l.retailer)} <i data-lucide="external-link" class="w-3.5 h-3.5"></i>
    </a>`).join('');

  function productCard(item, alternatives = []) {
    const meta = CATEGORY_META[item.category];
    const reasons = item.reasons.length ? `
      <ul class="space-y-1.5 my-4">
        ${item.reasons.map(r => `<li class="flex gap-2 text-sm text-gray-700"><i data-lucide="check" class="w-4 h-4 text-green-600 shrink-0 mt-0.5"></i><span>${esc(r)}</span></li>`).join('')}
      </ul>` : '';
    const alts = alternatives.length ? `
      <details class="mt-4 text-sm">
        <summary class="cursor-pointer text-green-800 font-semibold">Other good options</summary>
        <ul class="mt-2 space-y-2">
          ${alternatives.map(a => `
            <li class="flex flex-wrap items-center justify-between gap-2">
              <span>${esc(a.brand)} ${esc(a.name)} <span class="text-gray-500">~$${a.price}</span></span>
              <span class="flex gap-2">${linkButtons(a.links)}</span>
            </li>`).join('')}
        </ul>
      </details>` : '';
    return `
      <article class="bg-white rounded-xl border border-green-200 shadow-sm p-5 flex flex-col">
        <p class="text-xs font-bold uppercase tracking-wide text-green-700">${meta.emoji} ${meta.title}</p>
        <h3 class="text-lg font-bold text-gray-900 mt-1">${esc(item.brand)} ${esc(item.name)}</h3>
        <p class="text-sm text-gray-500">~$${item.price} · ${esc(item.material)}</p>
        <p class="text-sm text-gray-700 mt-3">${esc(item.summary)}</p>
        ${reasons}
        ${item.sizing ? `<p class="text-sm bg-green-50 rounded-lg px-3 py-2 text-green-900 flex gap-2"><i data-lucide="ruler" class="w-4 h-4 shrink-0 mt-0.5"></i><span>${esc(item.sizing)}</span></p>` : ''}
        <div class="flex flex-wrap gap-2 mt-4">${linkButtons(item.links)}</div>
        ${alts}
      </article>`;
  }

  function renderResult(r, data) {
    const breed = data.breed || 'your dog';
    const parts = [`For ${esc(breed)} (${data.weight_lbs} lbs, ${data.chest}" chest, ${data.neck}" neck)`];
    parts.push(`total ~$${r.total_price} · ${esc(r.budget.label)}`);
    let summary = parts.join(': ');
    if (r.budget.over_budget.length) {
      summary += `<br><span class="text-amber-700 text-sm">Nothing in the ${r.budget.over_budget.join(' / ')} category fits your dog within this budget, so we picked the most affordable option that fits.</span>`;
    }
    $('result-summary').innerHTML = summary;
    $('combo').innerHTML = ['harness', 'collar', 'leash']
      .filter(c => r.combo[c])
      .map(c => productCard(r.combo[c], r.alternatives[c] || []))
      .join('');
    $('extras').innerHTML = r.extras.map(e => productCard(e)).join('');
    show($('extras-wrap'), r.extras.length > 0);
    $('material-note').textContent = r.material_note;
    $('notes').innerHTML = r.notes.map(n => `<li>${esc(n)}</li>`).join('');
    show($('result'));
    icons();
    $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    $('result').focus({ preventScroll: true });
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const errorBox = $('form-error');
    const { data, missing } = collect();
    form.querySelectorAll('.is-missing').forEach(el => el.classList.remove('is-missing'));
    if (missing.length) {
      missing.forEach(el => el.classList.add('is-missing'));
      errorBox.textContent = `Please answer the ${missing.length} highlighted question${missing.length > 1 ? 's' : ''}.`;
      show(errorBox);
      missing[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }
    show(errorBox, false);
    const button = $('submit-btn');
    button.disabled = true;
    button.textContent = 'Finding the best gear…';
    try {
      const result = await api('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      renderResult(result, data);
    } catch (err) {
      errorBox.textContent = err.message;
      show(errorBox);
    } finally {
      button.disabled = false;
      button.textContent = 'Get Recommendation';
    }
  });

  form.addEventListener('change', e => e.target.closest('.is-missing')?.classList.remove('is-missing'));
  $('restart').addEventListener('click', () => form.scrollIntoView({ behavior: 'smooth' }));
})();
