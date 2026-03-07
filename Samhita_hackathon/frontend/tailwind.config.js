/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eaf3f9',
          100: '#cfe2ee',
          200: '#a8cede',
          300: '#7bb5cb',
          400: '#4d9bb8',
          500: '#1f5a7a',
          600: '#174966',
          700: '#0f2f45',
          800: '#0a1f30',
          900: '#051018',
        },
        surface: '#ffffff',
        'surface-alt': '#f6f9fb',
        border: '#d7e1e8',
        'text-strong': '#13293d',
        'text-muted': '#516477',
        accent: '#0f8b8d',
        danger: '#c43d3d',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
