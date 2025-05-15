FROM python:3.13-slim

# En cas de problème avec l'image 3.13-slim, vous pouvez utiliser en alternative:
# FROM python:3.13-rc-slim
# ou
# FROM python:3.12-slim

WORKDIR /app

# Copier uniquement les fichiers nécessaires pour installer les dépendances
# Cela permet de mieux utiliser le cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY . .

# Exposer le port sur lequel l'application s'exécute
EXPOSE 8080

# Commande pour démarrer l'application avec python directement
CMD ["python", "app.py"]