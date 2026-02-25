import { StudioClient } from "@/components/StudioClient";

export default function HomePage() {
  return (
    <main className="min-h-screen px-4 py-5 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <StudioClient />
        <footer className="mt-8 pb-2 text-center text-xs text-[var(--muted)]">
          Handwriting Studio | AI letter generation + dataset/training tooling
        </footer>
      </div>
    </main>
  );
}
