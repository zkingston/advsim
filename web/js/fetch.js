// The page's data files: a missing one is an error, not a parse failure of the 404 page.
export async function json(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}
