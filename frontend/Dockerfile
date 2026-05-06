FROM node:20-alpine AS base

FROM base AS builder
WORKDIR /app
COPY package.json yarn.lock* ./
# Omitimos la advertencia de dependencias, instalamos todo
RUN yarn install --frozen-lockfile || yarn install

COPY . .
# En Next.js 15, el build de producción
RUN yarn build

FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
