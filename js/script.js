/**
 * FreqHybrid Blog - Interactive Features
 */

document.addEventListener("DOMContentLoaded", function () {
  // Smooth scroll for TOC links
  const tocLinks = document.querySelectorAll(".toc a");
  tocLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault();
      const targetId = this.getAttribute("href");
      const targetSection = document.querySelector(targetId);

      if (targetSection) {
        targetSection.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });

        // Update URL without jumping
        history.pushState(null, null, targetId);
      }
    });
  });

  // Highlight current section in TOC
  const sections = document.querySelectorAll("section[id]");
  const tocItems = document.querySelectorAll(".toc a");

  window.addEventListener("scroll", function () {
    let current = "";

    sections.forEach((section) => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.clientHeight;
      if (pageYOffset >= sectionTop - 100) {
        current = section.getAttribute("id");
      }
    });

    tocItems.forEach((item) => {
      item.classList.remove("active");
      if (item.getAttribute("href") === `#${current}`) {
        item.classList.add("active");
      }
    });
  });

  // Add "Back to Top" button
  const backToTop = document.createElement("button");
  backToTop.innerHTML = "↑ Top";
  backToTop.className = "back-to-top";
  backToTop.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: #2E86AB;
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
        font-weight: bold;
        opacity: 0;
        transition: opacity 0.3s ease, transform 0.3s ease;
        z-index: 1000;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    `;

  document.body.appendChild(backToTop);

  window.addEventListener("scroll", function () {
    if (window.pageYOffset > 300) {
      backToTop.style.opacity = "1";
      backToTop.style.transform = "translateY(0)";
    } else {
      backToTop.style.opacity = "0";
      backToTop.style.transform = "translateY(10px)";
    }
  });

  backToTop.addEventListener("click", function () {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  });

  backToTop.addEventListener("mouseenter", function () {
    this.style.background = "#1e5f7a";
    this.style.transform = "translateY(-2px)";
  });

  backToTop.addEventListener("mouseleave", function () {
    this.style.background = "#2E86AB";
    this.style.transform = "translateY(0)";
  });

  // Image lazy loading fallback
  const images = document.querySelectorAll("img");
  images.forEach((img) => {
    img.addEventListener("error", function () {
      this.style.display = "none";
      const caption = this.nextElementSibling;
      if (caption && caption.classList.contains("caption")) {
        caption.innerHTML += " <em>(Image not available)</em>";
      }
    });
  });

  // Table responsive wrapper
  const tables = document.querySelectorAll("table");
  tables.forEach((table) => {
    if (!table.parentElement.classList.contains("table-responsive")) {
      const wrapper = document.createElement("div");
      wrapper.className = "table-responsive";
      wrapper.style.overflowX = "auto";
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    }
  });

  console.log("FreqHybrid Blog initialized ✓");
});
