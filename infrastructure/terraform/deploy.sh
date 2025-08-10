#!/bin/bash

# Script per deployare l'infrastruttura con Terraform

set -e

echo "🚀 Iniziando deployment dell'infrastruttura..."

# Verifica che Terraform sia installato
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform non è installato. Installa Terraform prima di continuare."
    exit 1
fi

# Verifica che gcloud sia configurato
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ gcloud non è autenticato. Esegui 'gcloud auth login' prima di continuare."
    exit 1
fi

# Verifica che il progetto sia impostato
PROJECT_ID=$(gcloud config get-value project)
if [ "$PROJECT_ID" != "mlops-breast-cancer" ]; then
    echo "❌ Progetto GCP non impostato correttamente. Imposta il progetto con 'gcloud config set project mlops-breast-cancer'"
    exit 1
fi

echo "✅ Progetto GCP configurato: $PROJECT_ID"

# Inizializza Terraform
echo "📦 Inizializzando Terraform..."
terraform init

# Verifica il piano
echo "📋 Verificando il piano di deployment..."
terraform plan -var-file="environments/dev.tfvars"

# Chiedi conferma
read -p "🤔 Procedere con il deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment annullato."
    exit 1
fi

# Applica il deployment
echo "🚀 Applicando il deployment..."
terraform apply -var-file="environments/dev.tfvars" -auto-approve

# Mostra gli outputs
echo "✅ Deployment completato! Outputs:"
terraform output

echo "🎉 Infrastruttura deployata con successo!"
