from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

from .engine import engine

# Create your views here.


@csrf_exempt
@require_http_methods(["POST"])
def upload(request):
    """
    Handle file upload and start background embedding process.
    Returns immediately - use /status/ to check progress.
    """
    try:
        file = request.FILES.get("file")

        if not file:
            return JsonResponse({"error": "No file provided"}, status=400)

        rows, columns = engine.ingest(file)

        return JsonResponse({
            "status": "processing",
            "message": "File uploaded. Embedding in progress...",
            "rows": rows,
            "columns": columns
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def status(request):
    """Check processing status."""
    return JsonResponse(engine.get_status())


@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    """
    Handle chat queries - routes to Text-to-Code or RAG based on query type.
    """
    try:
        data = json.loads(request.body)
        question = data.get("query", "")

        if not question:
            return JsonResponse({"error": "query parameter required"}, status=400)

        answer, method = engine.query(question)

        return JsonResponse({
            "answer": answer,
            "method": method
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def health(request):
    """Health check endpoint."""
    return JsonResponse({"status": "healthy"})
