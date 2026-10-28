"""Investigation assistant: drafts a cited case brief for one alert; never decides.

    case = build_case(alert, history, all_alerts)        # facts F* and signals S*
    result = draft_brief(case, provider, index)          # RAG over docs/sops + checks

The brief keeps recorded facts and model signals apart, cites SOP sections, and is checked
automatically (citations, numbers, decision language, context-only signals).
"""
