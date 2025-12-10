#!/bin/bash
set -e

# -----------------------------
# CONFIGURATION
# -----------------------------
NAMESPACE=openwhisk
NODEPORT=30080
SERVICE_NAME=ow-controller-nodeport

# -----------------------------
# 1. Detect master node
# -----------------------------
MASTER_NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
echo "Master node detected: $MASTER_NODE"

# -----------------------------
# 2. Check for taints and remove if present
# -----------------------------
TAINTS=$(kubectl describe node $MASTER_NODE | grep -i taints | awk -F: '{print $2}' | xargs)
if [ -n "$TAINTS" ]; then
    echo "Master node has taints: $TAINTS"
    echo "Removing taints so pods can schedule..."
    kubectl taint nodes $MASTER_NODE $TAINTS- || true
else
    echo "No taints on master node. Nothing to remove."
fi

# -----------------------------
# 3. Ensure NodePort service exists
# -----------------------------
EXISTS=$(kubectl get svc -n $NAMESPACE $SERVICE_NAME --ignore-not-found)
if [ -z "$EXISTS" ]; then
    echo "Creating NodePort service for controller..."
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: $SERVICE_NAME
  namespace: $NAMESPACE
spec:
  type: NodePort
  selector:
    app: owdev-openwhisk
    name: owdev-controller
  ports:
    - port: 8080
      targetPort: 8080
      protocol: TCP
      nodePort: $NODEPORT
EOF
else
    echo "NodePort service already exists."
fi

# -----------------------------
# 4. Fetch current OpenWhisk API key
# -----------------------------
if command -v wsk >/dev/null 2>&1; then
    WSK_AUTH=$(wsk property get --auth | awk '{print $4}')
    ENCODED_AUTH=$(echo -n "$WSK_AUTH" | base64)
    echo "Current API key retrieved successfully."
else
    echo "wsk CLI not found. Please install and configure wsk to get the API key."
fi

# -----------------------------
# 5. Display NodePort info
# -----------------------------
echo "NodePort setup complete."
echo "You can now invoke OpenWhisk actions via:"
echo "  http://127.0.0.1:$NODEPORT/api/v1/namespaces/_/actions/<ACTION_NAME>?blocking=true"
echo "Authorization header (Base64): $ENCODED_AUTH"

echo "Script finished!"
