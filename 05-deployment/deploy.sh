#!/bin/bash

set -e

echo "=========================================="
echo "🚀 DEPLOYMENT SCRIPT"
echo "=========================================="

# Step 1: Navigate to project
echo "📍 Navigating to 05-deployment..."
cd /workspaces/machine-learning-zoomcamp-homework/05-deployment

# Step 2: Verify files
echo "✅ Verifying files..."
if [ ! -f "predict.py" ]; then
    echo "❌ predict.py not found!"
    exit 1
fi
if [ ! -f "model.bin" ]; then
    echo "❌ model.bin not found!"
    exit 1
fi
echo "✅ All files present"

# Step 3: Commit changes
echo "\n📝 Committing to Git..."
cd /workspaces/machine-learning-zoomcamp-homework
git add 05-deployment/predict.py
git add 05-deployment/test.py
git commit -m "Update: Production-ready Churn Prediction API v3.0

IMPROVEMENTS:
- Fixed predict_proba list wrapping
- Added custom realistic example in Swagger UI
- Improved logging and monitoring
- Added business logic recommendations
- Better error handling
- Production-grade documentation" || echo "No changes to commit"

git push origin main
echo "✅ Pushed to GitHub"

# Step 4: Deploy to Fly.io
echo "\n🌐 Deploying to Fly.io..."
cd /workspaces/machine-learning-zoomcamp-homework/05-deployment
flyctl deploy

echo "\n=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo "🔗 API URL: https://red-butterfly-7700.fly.dev"
echo "📚 Docs: https://red-butterfly-7700.fly.dev/docs"
echo "💚 Health: https://red-butterfly-7700.fly.dev/health"
echo "=========================================="
