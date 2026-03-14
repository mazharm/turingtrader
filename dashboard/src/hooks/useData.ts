import { useState, useEffect } from 'react';

interface UseDataResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

const BASE = import.meta.env.BASE_URL;

export function useData<T>(path: string): UseDataResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(`${BASE}data/${path}`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
        return res.json();
      })
      .then((json: T) => {
        if (!cancelled) {
          setData(json);
          setLoading(false);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [path]);

  return { data, loading, error };
}
