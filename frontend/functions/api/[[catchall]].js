export async function onRequest(context) {
  const url = new URL(context.request.url);
  const targetUrl = `https://whysooraj-funforge-backend.hf.space${url.pathname}${url.search}`;
  const requestInit = {
    method: context.request.method,
    headers: context.request.headers,
  };
  if (context.request.method !== "GET" && context.request.method !== "HEAD") {
    requestInit.body = context.request.body;
  }
  return fetch(targetUrl, requestInit);
}
