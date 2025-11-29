// Quorum-MCP Documentation - Main JavaScript

(function() {
  'use strict';

  // ============================================
  // Theme Toggle
  // ============================================
  function initTheme() {
    const themeToggle = document.querySelector('.theme-toggle');
    if (!themeToggle) return;

    // Get stored theme or system preference
    const storedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = storedTheme || (systemPrefersDark ? 'dark' : 'light');

    document.documentElement.setAttribute('data-theme', initialTheme);

    themeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
    });

    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (!localStorage.getItem('theme')) {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      }
    });
  }

  // ============================================
  // Mobile Menu Toggle
  // ============================================
  function initMobileMenu() {
    const menuToggle = document.querySelector('.mobile-menu-toggle');
    const mainNav = document.querySelector('.main-nav');

    if (!menuToggle || !mainNav) return;

    menuToggle.addEventListener('click', () => {
      const isExpanded = menuToggle.getAttribute('aria-expanded') === 'true';
      menuToggle.setAttribute('aria-expanded', !isExpanded);
      mainNav.classList.toggle('is-open');
      document.body.classList.toggle('menu-open');
    });

    // Close menu on escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && mainNav.classList.contains('is-open')) {
        menuToggle.setAttribute('aria-expanded', 'false');
        mainNav.classList.remove('is-open');
        document.body.classList.remove('menu-open');
      }
    });
  }

  // ============================================
  // Table of Contents Generator
  // ============================================
  function initTableOfContents() {
    const tocContainer = document.getElementById('toc');
    const content = document.querySelector('.content');

    if (!tocContainer || !content) return;

    const headings = content.querySelectorAll('h2, h3');
    if (headings.length === 0) {
      tocContainer.parentElement.style.display = 'none';
      return;
    }

    const toc = document.createElement('ul');

    headings.forEach((heading, index) => {
      // Add ID if not present
      if (!heading.id) {
        heading.id = 'heading-' + index;
      }

      const li = document.createElement('li');
      li.className = heading.tagName.toLowerCase();

      const link = document.createElement('a');
      link.href = '#' + heading.id;
      link.textContent = heading.textContent;

      li.appendChild(link);
      toc.appendChild(li);
    });

    tocContainer.appendChild(toc);

    // Highlight active section on scroll
    const observerOptions = {
      rootMargin: '-80px 0px -80% 0px',
      threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const link = tocContainer.querySelector(`a[href="#${entry.target.id}"]`);
        if (link) {
          if (entry.isIntersecting) {
            tocContainer.querySelectorAll('a').forEach(a => a.classList.remove('active'));
            link.classList.add('active');
          }
        }
      });
    }, observerOptions);

    headings.forEach(heading => observer.observe(heading));
  }

  // ============================================
  // Smooth Scroll for Anchor Links
  // ============================================
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function(e) {
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;

        const target = document.querySelector(targetId);
        if (target) {
          e.preventDefault();
          const headerOffset = 80;
          const elementPosition = target.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

          window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
          });

          // Update URL without jumping
          history.pushState(null, null, targetId);
        }
      });
    });
  }

  // ============================================
  // Code Copy Button
  // ============================================
  function initCodeCopy() {
    const codeBlocks = document.querySelectorAll('pre');

    codeBlocks.forEach(block => {
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';
      block.parentNode.insertBefore(wrapper, block);
      wrapper.appendChild(block);

      const copyButton = document.createElement('button');
      copyButton.className = 'copy-button';
      copyButton.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
          <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
        </svg>
        <span>Copy</span>
      `;

      copyButton.addEventListener('click', async () => {
        const code = block.querySelector('code');
        const text = code ? code.textContent : block.textContent;

        try {
          await navigator.clipboard.writeText(text);
          copyButton.classList.add('copied');
          copyButton.querySelector('span').textContent = 'Copied!';

          setTimeout(() => {
            copyButton.classList.remove('copied');
            copyButton.querySelector('span').textContent = 'Copy';
          }, 2000);
        } catch (err) {
          console.error('Failed to copy:', err);
        }
      });

      wrapper.appendChild(copyButton);
    });

    // Add styles for copy button
    const style = document.createElement('style');
    style.textContent = `
      .code-block-wrapper {
        position: relative;
      }
      .copy-button {
        position: absolute;
        top: 8px;
        right: 8px;
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 4px 8px;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        color: #94a3b8;
        font-size: 12px;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.2s, background 0.2s;
      }
      .code-block-wrapper:hover .copy-button {
        opacity: 1;
      }
      .copy-button:hover {
        background: rgba(255, 255, 255, 0.2);
        color: #e2e8f0;
      }
      .copy-button.copied {
        color: #10b981;
      }
    `;
    document.head.appendChild(style);
  }

  // ============================================
  // External Link Handling
  // ============================================
  function initExternalLinks() {
    document.querySelectorAll('a[href^="http"]').forEach(link => {
      if (!link.href.includes(window.location.hostname)) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
      }
    });
  }

  // ============================================
  // Initialize All Features
  // ============================================
  function init() {
    initTheme();
    initMobileMenu();
    initTableOfContents();
    initSmoothScroll();
    initCodeCopy();
    initExternalLinks();
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
