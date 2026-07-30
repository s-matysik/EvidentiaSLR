<?php

/**
 * Place at: core-backend/database/migrations/
 *
 * One row per (review, stage, query). The certificate is stored in full
 * rather than reduced to the ranked list, because a ranked list without its
 * provenance cannot be verified later — which is the whole point.
 *
 * `evidence_digest` is indexed so that two reviews that arrived at the same
 * evidence set can be identified, and so a re-run can be checked against a
 * stored digest without deserialising the JSON.
 */

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('review_evidence_sets', function (Blueprint $table) {
            $table->id();
            $table->foreignId('review_id')->constrained('reviews')->cascadeOnDelete();
            $table->string('stage', 50);
            $table->text('query');
            $table->char('query_hash', 64);
            $table->char('corpus_hash', 64)->nullable();
            $table->unsignedInteger('corpus_size')->default(0);
            $table->char('evidence_digest', 64)->nullable();
            $table->json('certificate')->nullable();
            $table->json('evidence')->nullable();
            $table->string('status', 30)->default('pending');
            $table->text('error_message')->nullable();
            $table->timestamps();

            $table->unique(['review_id', 'stage', 'query_hash']);
            $table->index('review_id');
            $table->index('stage');
            $table->index('status');
            $table->index('evidence_digest');
        });

        // Verification attempts, so a review can show a reproducibility
        // history rather than a single snapshot.
        Schema::create('review_evidence_verifications', function (Blueprint $table) {
            $table->id();
            $table->foreignId('review_evidence_set_id')
                ->constrained('review_evidence_sets')
                ->cascadeOnDelete();
            $table->foreignId('verified_by_user_id')->nullable()
                ->constrained('users')->nullOnDelete();
            $table->boolean('ok');
            $table->json('mismatches')->nullable();
            $table->json('detail')->nullable();
            $table->timestamp('verified_at');
            $table->timestamps();

            $table->index('review_evidence_set_id');
            $table->index('ok');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('review_evidence_verifications');
        Schema::dropIfExists('review_evidence_sets');
    }
};
