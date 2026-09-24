/**
 * DOM Helper Utilities
 */

export const $ = (id) => document.getElementById(id);
export const $$ = (sel) => document.querySelectorAll(sel);

export function safeText(el, text) {
  if (el) el.textContent = text;
}

export function safeHtml(el, html) {
  if (el) el.innerHTML = html;
}

export function safeClass(el, className, force) {
  if (el) el.classList.toggle(className, force);
}
