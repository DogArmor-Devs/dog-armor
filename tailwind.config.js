/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/templates/**/*.html", "./app/static/js/**/*.js"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', "-apple-system", "BlinkMacSystemFont", '"Segoe UI"', "Roboto", "Helvetica", "Arial", "sans-serif"],
        serif: ["Newsreader", "Georgia", '"Times New Roman"', "serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        canvas: "#f8fafc",
        ink: "#1f2937",
        body: "#475467",
        muted: "#667085",
        line: "#e4e7ec",
        night: "#0a0b0d",
        accent: { DEFAULT: "#0cc7ed", 50: "#e6f9fd", 100: "#c9f2fb", 600: "#0aa8c9", 700: "#0889a4" },
      },
      boxShadow: {
        soft: "0 1px 2px rgba(16,24,40,.04), 0 8px 24px -8px rgba(16,24,40,.08)",
        lift: "0 24px 48px -16px rgba(16,24,40,.18)",
        glow: "0 6px 20px -6px rgba(12,199,237,.55)",
      },
      letterSpacing: { display: "-0.02em" },
    },
  },
  plugins: [],
};
