export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (
      url.pathname.startsWith("/api/") ||
      url.pathname.startsWith("/sample_sketches/") ||
      url.pathname.startsWith("/outputs/") ||
      url.pathname.startsWith("/uploads/")
    ) {
      const targetUrl = new URL(
        url.pathname + url.search,
        "https://whysooraj-funforge-backend.hf.space"
      );
      return fetch(new Request(targetUrl, request));
    }
    return env.ASSETS.fetch(request);
  },
};
