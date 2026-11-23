# AlertaFuego: frontend

Aplicación web de AlertaFuego, hecha con Nuxt 4 y Nuxt UI. Muestra las detecciones de incendios sobre un mapa de Argentina, con filtros, el detalle de cada foco y los reportes de los usuarios; las estadísticas por provincia, departamento y zona; las zonas de alerta de cada usuario y la configuración de sus avisos. Los datos vienen de la API del [backend](../backend), y el inicio de sesión, de Supabase.

## Cómo correrlo

Requisitos: Node 22 y un proyecto de Supabase (el mismo que usa el backend).

```bash
cp .env.example .env    # completar la URL y la clave pública de Supabase y la URL de la API
npm ci
npm run dev             # http://localhost:3000
```

Con el backend corriendo en local, `NUXT_PUBLIC_API_BASE=http://localhost:8000`.

Para generar el sitio estático que se publica en S3 y CloudFront:

```bash
npm run generate        # genera .output/public
```

El despliegue en AWS está descrito en [`terraform/README.md`](terraform/README.md).

## Cómo correr las pruebas

```bash
npm run test            # Vitest: utilidades, estado, composables y componentes
npm run test:coverage   # lo mismo, con cobertura
npx playwright install chromium   # solo la primera vez
npm run test:e2e        # Playwright: navegación, mapa, accesibilidad y vista en teléfono
```

Las pruebas de Playwright generan el sitio y lo sirven en el puerto 3100. No usan el backend ni Supabase: `test/e2e/fixtures.ts` responde cada pedido con las formas del contrato de la API.
