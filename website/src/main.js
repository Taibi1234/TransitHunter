document.documentElement.classList.add('has-js');

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

// Deterministic schematic trace. No scientific measurements are used here.
function buildLightCurve() {
  const path = document.querySelector('#light-curve-path');
  const points = document.querySelector('#light-curve-points');
  if (!path || !points) return;

  const width = 652;
  const left = 44;
  const baseline = 154;
  const samples = 560;
  const period = 0.8374290408;
  let seed = 11904151;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  const coordinates = [];
  for (let index = 0; index <= samples; index += 1) {
    const time = (index / samples) * 4;
    const phase = ((time - 0.31 + period / 2) % period) - period / 2;
    const transit = 82 * Math.exp(-0.5 * (phase / 0.029) ** 2);
    const shoulder = 10 * Math.exp(-0.5 * (phase / 0.055) ** 2);
    const noise = (random() - 0.5) * 13 + Math.sin(index * 0.21) * 2;
    const x = left + (index / samples) * width;
    const y = baseline + transit + shoulder + noise;
    coordinates.push([x, y]);
  }

  path.setAttribute('d', coordinates.map(([x, y], index) => `${index === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`).join(' '));
  const namespace = 'http://www.w3.org/2000/svg';
  coordinates.filter((_, index) => index % 17 === 0).forEach(([x, y]) => {
    const point = document.createElementNS(namespace, 'circle');
    point.setAttribute('cx', x.toFixed(2));
    point.setAttribute('cy', y.toFixed(2));
    point.setAttribute('r', '1.65');
    points.append(point);
  });

  if (!reducedMotion.matches) path.classList.add('draw');
}

buildLightCurve();

const header = document.querySelector('[data-header]');
const menuButton = document.querySelector('[data-menu-toggle]');
const navigation = document.querySelector('[data-nav]');

function updateHeader() {
  header?.classList.toggle('scrolled', window.scrollY > 24);
}

function closeMenu() {
  if (!menuButton || !navigation) return;
  menuButton.setAttribute('aria-expanded', 'false');
  menuButton.setAttribute('aria-label', 'Open navigation');
  navigation.classList.remove('open');
  header?.classList.remove('menu-active');
  document.body.classList.remove('menu-open');
}

function toggleMenu() {
  if (!menuButton || !navigation) return;
  const opening = menuButton.getAttribute('aria-expanded') !== 'true';
  menuButton.setAttribute('aria-expanded', String(opening));
  menuButton.setAttribute('aria-label', opening ? 'Close navigation' : 'Open navigation');
  navigation.classList.toggle('open', opening);
  header?.classList.toggle('menu-active', opening);
  document.body.classList.toggle('menu-open', opening);
}

updateHeader();
window.addEventListener('scroll', updateHeader, { passive: true });
menuButton?.addEventListener('click', toggleMenu);
navigation?.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));
window.addEventListener('resize', () => {
  if (window.innerWidth > 900) closeMenu();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && menuButton?.getAttribute('aria-expanded') === 'true') {
    closeMenu();
    menuButton.focus();
  }
});

// Reveal content only when it approaches the viewport; show everything otherwise.
const reveals = [...document.querySelectorAll('.reveal')];
if (reducedMotion.matches || !('IntersectionObserver' in window)) {
  reveals.forEach((element) => element.classList.add('is-visible'));
} else {
  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { rootMargin: '0px 0px -6% 0px', threshold: 0.06 });
  reveals.forEach((element) => revealObserver.observe(element));
}

const sectionLinks = [...document.querySelectorAll('.primary-nav a[href^="#"]')];
const observedSections = sectionLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);

if ('IntersectionObserver' in window) {
  const sectionObserver = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    sectionLinks.forEach((link) => {
      const active = link.getAttribute('href') === `#${visible.target.id}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }, { rootMargin: '-25% 0px -55% 0px', threshold: [0, 0.1, 0.3] });
  observedSections.forEach((section) => sectionObserver.observe(section));
}

const toast = document.querySelector('[data-toast]');
let toastTimer;
function showToast(message) {
  if (!toast) return;
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.add('visible');
  toastTimer = window.setTimeout(() => toast.classList.remove('visible'), 2400);
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.append(textarea);
  textarea.select();
  const successful = document.execCommand('copy');
  textarea.remove();
  if (!successful) throw new Error('Copy unavailable');
}

const copyButton = document.querySelector('[data-copy-command]');
copyButton?.addEventListener('click', async () => {
  const command = document.querySelector('#run-command')?.textContent.trim();
  if (!command) return;
  try {
    await copyText(command);
    const label = copyButton.querySelector('[data-copy-label]');
    if (label) label.textContent = 'Copied';
    showToast('Reproduction command copied.');
    window.setTimeout(() => { if (label) label.textContent = 'Copy'; }, 2200);
  } catch {
    showToast('Copy unavailable. Select the command manually.');
  }
});

const figures = {
  bls: {
    src: './assets/kepler10-bls.png',
    alt: 'Expanded BLS periodogram and phase-folded Kepler-10 blind recovery diagnostic.',
    caption: 'Figure 01 · Blind BLS recovery · real pipeline output'
  },
  injection: {
    src: './assets/injection-recovery.png',
    alt: 'Expanded nine-trial injection and recovery benchmark, with one missed 100 ppm injection.',
    caption: 'Figure 02 · Injection/recovery · real pipeline output'
  }
};
const dialog = document.querySelector('[data-figure-dialog]');
const dialogImage = dialog?.querySelector('[data-dialog-image]');
const dialogCaption = dialog?.querySelector('#dialog-caption');

function closeDialog() {
  if (dialog?.open) dialog.close();
}

document.querySelectorAll('[data-figure]').forEach((button) => {
  button.addEventListener('click', () => {
    const figure = figures[button.dataset.figure];
    if (!dialog || !dialogImage || !dialogCaption || !figure) return;
    dialogImage.src = figure.src;
    dialogImage.alt = figure.alt;
    dialogCaption.textContent = figure.caption;
    dialog.showModal();
    document.body.classList.add('dialog-open');
  });
});
dialog?.querySelector('[data-dialog-close]')?.addEventListener('click', closeDialog);
dialog?.addEventListener('click', (event) => {
  if (event.target === dialog) closeDialog();
});
dialog?.addEventListener('close', () => document.body.classList.remove('dialog-open'));
