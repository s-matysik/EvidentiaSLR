<?php

/**
 * Place at: core-backend/src/Reviews/Jobs/RetrieveReviewEvidenceJob.php
 *
 * Mirrors ExtractReviewSourceFileJob, with the defects found in that job
 * fixed rather than copied:
 *
 *  - config() instead of env(), so the job survives `php artisan config:cache`;
 *  - $tries / backoff() / failed(), so a transient failure retries and a
 *    permanent one lands in failed_jobs instead of being swallowed;
 *  - ShouldBeUnique, so a double-click cannot run two retrievals for the
 *    same review and stage concurrently.
 *
 * Requires config/services.php:
 *
 *   'evidentia' => [
 *       'url'     => env('EVIDENTIA_URL', 'http://evidentia:8000'),
 *       'timeout' => env('EVIDENTIA_TIMEOUT', 900),
 *   ],
 */

namespace LitRev\CoreBackend\Reviews\Jobs;

use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldBeUnique;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use LitRev\CoreBackend\Reviews\Models\Review;
use LitRev\CoreBackend\Reviews\Models\ReviewEvidenceSet;
use Throwable;

class RetrieveReviewEvidenceJob implements ShouldQueue, ShouldBeUnique
{
    use Dispatchable;
    use InteractsWithQueue;
    use Queueable;
    use SerializesModels;

    public int $tries = 3;
    public int $timeout = 960;

    public function __construct(
        public int $reviewId,
        public string $query,
        public string $stage = 'data_extraction',
        public array $options = [],
    ) {
    }

    public function uniqueId(): string
    {
        return $this->reviewId . ':' . $this->stage . ':' . sha1($this->query);
    }

    public function backoff(): array
    {
        return [10, 60, 180];
    }

    public function handle(): void
    {
        $review = Review::query()->with('sourceFiles')->find($this->reviewId);

        if (! $review) {
            return;
        }

        $sourceFiles = $review->sourceFiles
            ->where('extraction_status', 'extracted')
            ->map(fn ($file) => [
                'extraction_status' => $file->extraction_status,
                'extracted_text' => $file->extracted_text,
                'extracted_metadata' => $file->extracted_metadata ?? [],
            ])
            ->values()
            ->all();

        if ($sourceFiles === []) {
            throw new \RuntimeException('No extracted source files available for review ' . $review->id);
        }

        $response = Http::timeout((int) config('services.evidentia.timeout', 900))
            ->acceptJson()
            ->post(
                rtrim((string) config('services.evidentia.url'), '/') . '/retrieve',
                array_merge([
                    'query' => $this->query,
                    'source_files' => $sourceFiles,
                    'k' => 20,
                    'mode' => 'hybrid',
                    'chunker' => 'tei-section',
                    'index' => 'flat',
                    'embedder' => 'sbert',
                ], $this->options),
            );

        $response->throw();

        $payload = $response->json();

        ReviewEvidenceSet::updateOrCreate(
            [
                'review_id' => $review->id,
                'stage' => $this->stage,
                'query_hash' => hash('sha256', $this->query),
            ],
            [
                'query' => $this->query,
                'corpus_hash' => $payload['corpus']['hash'],
                'corpus_size' => $payload['corpus']['size'],
                'evidence_digest' => $payload['certificate']['evidence']['digest'],
                'certificate' => $payload['certificate'],
                'evidence' => $payload['evidence'],
                'status' => 'ready',
                'error_message' => null,
            ],
        );
    }

    public function failed(Throwable $exception): void
    {
        Log::error('Evidence retrieval failed', [
            'review_id' => $this->reviewId,
            'stage' => $this->stage,
            'message' => $exception->getMessage(),
        ]);

        ReviewEvidenceSet::updateOrCreate(
            [
                'review_id' => $this->reviewId,
                'stage' => $this->stage,
                'query_hash' => hash('sha256', $this->query),
            ],
            [
                'query' => $this->query,
                'status' => 'failed',
                'error_message' => $exception->getMessage(),
            ],
        );
    }
}
