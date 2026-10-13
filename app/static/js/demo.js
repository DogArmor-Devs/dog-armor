(() => {
  const form = document.getElementById('demo-form');
  const fileInput = document.getElementById('dog_image');
  const dropzone = document.getElementById('dropzone');
  const breedInput = document.getElementById('breed');
  const breedNames = new Set([...document.querySelectorAll('#breed-options option')].map(o => o.value.toLowerCase()));
  const MEASURES = ['weight_lbs', 'neck', 'chest', 'belly', 'back'];
  const YES_NO = ['pulling', 'working', 'sports', 'growing', 'heat_sensitive', 'hiking', 'hunting', 'water', 'cleaning_experience'];
  const CHOICES = ['aggression', 'training', 'activity', 'climate', 'budget'];
  const GEAR = ['harness', 'collar', 'leash'];
  const budgetCaps = JSON.parse(document.querySelector('[data-caps]').dataset.caps);
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
      input.title = ranges[field] ? `Typical range: ${ranges[field][0]} to ${ranges[field][1]}` : '';
      setBadge(field, 'estimate');
    }
  }

  function applyTraits(newTraits, breed) {
    traits = newTraits;
    const hint = $('heat-hint');
    if (!traits || !(traits.brachycephalic || traits.coat === 'thick')) { show(hint, false); return; }
    const why = traits.brachycephalic ? 'flat-faced' : 'thick-coated';
    hint.textContent = `We picked "Yes" because ${breed}s are ${why} and overheat easily.`;
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

  // ---------- gear choice and budget ----------
  const selectedGear = () => GEAR.filter(g => form.querySelector(`input[name="gear"][value="${g}"]`).checked);
  const joinGear = list => list.length < 3 ? list.join(' and ') : `${list.slice(0, -1).join(', ')}, and ${list[list.length - 1]}`;

  function updateBudget() {
    const gear = selectedGear();
    const counted = gear.length ? gear : GEAR;
    const total = tier => counted.reduce((sum, g) => sum + budgetCaps[tier][g], 0);
    $('budget-title').textContent = gear.length ? `Budget for ${joinGear(gear)}` : 'Budget';
    $('budget-hint').textContent = gear.length === 0
      ? 'Pick your gear in step 1 to see ranges for it.'
      : gear.length === 1 ? `Price range for the ${gear[0]}.` : `Price range for the ${joinGear(gear)} together.`;
    document.querySelector('[data-tier-label="low"]').textContent = `Under $${total('low')}`;
    document.querySelector('[data-tier-label="medium"]').textContent = `$${total('low')} to $${total('medium')}`;
  }
  form.querySelectorAll('input[name="gear"]').forEach(input => input.addEventListener('change', updateBudget));
  updateBudget();

  // ---------- photo analysis ----------
  function renderBreeds(breeds, isDog) {
    const list = $('breed-list');
    list.innerHTML = breeds.map((b, i) => `
      <li>
        <button type="button" class="breed-choice ${i === 0 && isDog ? 'is-active' : ''}" data-breed="${esc(b.name)}">
          <span class="flex items-center gap-3 text-sm">
            <span class="font-mono text-xs text-muted">0${i + 1}</span>
            <span class="flex-1 text-ink">${esc(b.name)}</span>
            <span class="font-semibold ${i === 0 && isDog ? 'text-emerald-600' : 'text-muted'}">${Math.round(b.confidence * 100)}%</span>
          </span>
          <span class="block h-1.5 bg-gray-100 rounded-full mt-2 overflow-hidden">
            <span class="block h-1.5 ${i === 0 && isDog ? 'bg-emerald-500' : 'bg-gray-300'} rounded-full" style="width:${Math.max(2, Math.round(b.confidence * 100))}%"></span>
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
        $('analysis-error').textContent = "We couldn't find a dog in this photo. Try a clearer one, or pick the breed and enter measurements below.";
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
    const data = { breed: breedInput.value.trim(), gear: selectedGear() };
    const missing = [];   // elements to highlight
    if (!data.gear.length) missing.push($('gear-group'));
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
    harness: { title: 'Harness', color: 'bg-orange-500', icon: 'shield' },
    collar: { title: 'Collar', color: 'bg-emerald-500', icon: 'circle-dashed' },
    leash: { title: 'Leash', color: 'bg-accent', icon: 'link' },
    extra: { title: 'Extra', color: 'bg-violet-500', icon: 'plus' },
  };

  const linkButtons = links => links.map(l => `
    <a href="${esc(l.url)}" target="_blank" rel="noopener noreferrer" class="buy-link">
      ${esc(l.retailer)} <i data-lucide="arrow-up-right" class="w-3.5 h-3.5"></i>
    </a>`).join('');

  function productCard(item, alternatives = []) {
    const meta = CATEGORY_META[item.category];
    const reasons = item.reasons.length ? `
      <ul class="mt-5 space-y-2.5">
        ${item.reasons.map(r => `<li class="flex gap-2.5 text-sm text-body"><i data-lucide="check" class="w-4 h-4 text-emerald-500 shrink-0 mt-0.5"></i><span>${esc(r)}</span></li>`).join('')}
      </ul>` : '';
    const alts = alternatives.length ? `
      <details class="mt-5 border-t border-line pt-4 text-sm">
        <summary class="cursor-pointer list-none flex items-center justify-between text-ink font-medium">Other good options <i data-lucide="plus" class="w-4 h-4 text-gray-400"></i></summary>
        <ul class="mt-3 space-y-3">
          ${alternatives.map(a => `
            <li class="flex flex-wrap items-center justify-between gap-2">
              <span class="text-body">${esc(a.brand)} ${esc(a.name)} <span class="font-mono text-xs text-muted">$${a.price}</span></span>
              <span class="flex gap-2">${linkButtons(a.links)}</span>
            </li>`).join('')}
        </ul>
      </details>` : '';
    return `
      <article class="panel-soft p-6 flex flex-col">
        <div class="flex items-center justify-between">
          <span class="flex items-center gap-2.5 text-sm font-medium text-ink"><span class="icon-tile !w-8 !h-8 !rounded-lg ${meta.color}"><i data-lucide="${meta.icon}" class="w-4 h-4"></i></span>${meta.title}</span>
          <span class="font-mono text-sm text-ink">$${item.price}</span>
        </div>
        <h3 class="mt-5 text-xl font-semibold text-ink leading-snug">${esc(item.brand)} ${esc(item.name)}</h3>
        <p class="mt-1 text-xs text-muted">${esc(item.material)}</p>
        <p class="mt-3 text-[15px] text-muted leading-relaxed">${esc(item.summary)}</p>
        ${reasons}
        ${item.sizing ? `<p class="mt-5 flex gap-2 rounded-xl bg-accent-50 px-3.5 py-2.5 text-sm text-ink"><i data-lucide="ruler" class="w-4 h-4 shrink-0 mt-0.5 text-accent-700"></i><span>${esc(item.sizing)}</span></p>` : ''}
        <div class="mt-auto pt-5 flex flex-wrap gap-2">${linkButtons(item.links)}</div>
        ${alts}
      </article>`;
  }

  function renderResult(r, data) {
    $('result-dog').textContent = data.breed || 'your dog';
    let summary = `${data.weight_lbs} lb, ${data.chest} inch chest, ${data.neck} inch neck. Total about $${r.total_price}. Budget: ${esc(r.budget.label)}.`;
    if (r.budget.over_budget.length) {
      summary += `<br><span class="text-amber-700 text-sm">No ${r.budget.over_budget.join(' or ')} fits your dog at this budget, so we picked the most affordable one that does.</span>`;
    }
    $('result-summary').innerHTML = summary;
    $('combo').innerHTML = ['harness', 'collar', 'leash']
      .filter(c => r.combo[c])
      .map(c => productCard(r.combo[c], r.alternatives[c] || []))
      .join('');
    $('extras').innerHTML = r.extras.map(e => productCard(e)).join('');
    show($('extras-wrap'), r.extras.length > 0);
    $('material-note').textContent = r.material_note;
    $('notes').innerHTML = r.notes.map(n => `<li class="flex gap-2.5"><span class="mt-2 h-1.5 w-1.5 rounded-full bg-gray-300 shrink-0"></span><span>${esc(n)}</span></li>`).join('');
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
    button.textContent = 'Finding the right gear…';
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
      button.textContent = 'Get my recommendation';
    }
  });

  form.addEventListener('change', e => e.target.closest('.is-missing')?.classList.remove('is-missing'));
  $('restart').addEventListener('click', () => form.scrollIntoView({ behavior: 'smooth' }));
})();
