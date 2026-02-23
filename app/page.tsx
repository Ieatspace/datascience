import { StudioClient } from "@/components/StudioClient";

export default function HomePage() {
  return (
    <main className="relative min-h-screen overflow-hidden px-4 py-6 sm:px-6 lg:px-8">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute inset-x-0 top-[-20%] mx-auto h-[34rem] w-[34rem] rounded-full bg-[radial-gradient(circle_at_center,rgba(236,225,202,0.7),rgba(236,225,202,0))] blur-3xl" />
        <div className="absolute right-[-8%] top-[14%] h-[26rem] w-[26rem] rounded-full bg-[radial-gradient(circle_at_center,rgba(205,223,239,0.55),rgba(205,223,239,0))] blur-3xl" />
        <div className="absolute left-[-10%] bottom-[-5%] h-[22rem] w-[22rem] rounded-full bg-[radial-gradient(circle_at_center,rgba(214,221,206,0.45),rgba(214,221,206,0))] blur-3xl" />
        <div className="noise-overlay absolute inset-0 opacity-40" />
      </div>

      <div className="relative mx-auto max-w-7xl">
        <div className="mb-5 rounded-2xl border border-white/60 bg-white/50 p-4 shadow-sm backdrop-blur-sm noise-overlay sm:p-5">
          <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            <div>
              <p className="section-chip w-fit">Handwrite Studio</p>
              <h2 className="mt-3 text-2xl font-semibold text-slate-900 sm:text-3xl">
                A polished handwriting generation workspace
              </h2>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
                Create previews, compare history, and export images in a clean
                production-style interface while the handwriting engine continues to
                improve.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3">
              <div className="rounded-xl border border-border/70 bg-white/75 p-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
                  Status
                </p>
                <p className="mt-1 text-sm font-semibold text-foreground">
                  In Development
                </p>
              </div>
              <div className="rounded-xl border border-border/70 bg-white/75 p-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
                  Workflow
                </p>
                <p className="mt-1 text-sm font-semibold text-foreground">
                  Generate + Review
                </p>
              </div>
              <div className="rounded-xl border border-border/70 bg-white/75 p-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
                  UX
                </p>
                <p className="mt-1 text-sm font-semibold text-foreground">
                  History + Export
                </p>
              </div>
            </div>
          </div>
        </div>

        <StudioClient />

        <footer className="mt-8 pb-2 text-center text-xs text-muted-foreground">
          Handwrite Studio | polished UI experience with an evolving handwriting
          engine
        </footer>
      </div>
    </main>
  );
}
