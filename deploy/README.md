# Despliegue en la nube (docker-compose sobre un VPS)

Producción usa **el mismo stack de 2 contenedores** que local (cumple la pauta
del curso también en la nube). No hay imagen única ni workarounds: se corre el
`docker-compose` real en un VPS/VM.

## Requisitos

- Un VPS/VM Linux con IP pública (Hetzner, DigitalOcean, Oracle Free Tier,
  AWS EC2, etc.). 1 vCPU / 2 GB RAM alcanzan (los modelos son pequeños, CPU).
- Puerto **80** abierto en el firewall.

## Pasos

1. **Instalar Docker + Compose** en el VPS (Ubuntu/Debian):

   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER   # reabrir sesión tras esto
   ```

2. **Clonar el repo** (los artefactos entrenados vienen versionados, no hay que
   entrenar en el servidor):

   ```bash
   git clone <URL_DEL_REPO> leadscore && cd leadscore
   ```

3. **Levantar el stack** en el puerto 80:

   ```bash
   docker compose -f docker-compose.prod.yml up -d --build
   ```

4. **Verificar**:

   ```bash
   curl -s http://localhost/api/health      # {"status":"ok","models_loaded":true,...}
   ```

   La app queda en `http://<IP_DEL_VPS>/`.

## Dominio + HTTPS (opcional)

Apuntar un dominio A-record a la IP y poner un reverse proxy con TLS automático
(por ejemplo Caddy) delante del `frontend`, o terminar TLS en el propio nginx.

## Operación

```bash
docker compose -f docker-compose.prod.yml ps       # estado
docker compose -f docker-compose.prod.yml logs -f  # logs
docker compose -f docker-compose.prod.yml down      # detener
git pull && docker compose -f docker-compose.prod.yml up -d --build  # actualizar
```
