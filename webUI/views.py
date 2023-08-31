from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import sys

sys.path.append("..")

FIRST_TIME = True


def index(request):
    from main import warm_up_for_web

    global FIRST_TIME
    if FIRST_TIME:
        FIRST_TIME = False
        warm_up_for_web()
    index_file = open("webUI/html/index.html", "r", encoding="utf-8").read()
    return HttpResponse(index_file)


def search(request):
    query = request.GET.get("query")
    print(query)
    from main import ui_evaluate

    results = ui_evaluate(query)
    return render(
        request,
        "/Users/dingluran/Projects/网页信息检索/webUI/html/search_results.html",
        {"results": results},
    )


def track_click(request):
    if request.method == "POST":
        url = request.POST.get("url")
        print(url)
        return JsonResponse({"status": "ok"})
    else:
        return JsonResponse({"status": "error", "error": "Invalid request method"})
