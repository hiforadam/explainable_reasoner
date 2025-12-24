"""Main CLI interface."""
import argparse
import json
import logging
import sys
from reasoner.data import load_examples, load_token_descriptions, write_jsonl
from reasoner.tokenizer import ClosedVocabTokenizer
from reasoner.train import train_reasoner, train_dual
from reasoner.model import ReasonerArtifacts
from reasoner.selector import SelectorArtifacts
from reasoner.generate import generate, generate_dual
from reasoner.config import ModelConfig, default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def cmd_build_token_desc(args):
    """Build token descriptions scaffold."""
    try:
        ex = load_examples(args.data)
        if not ex:
            logger.error(f"No examples found in {args.data}")
            sys.exit(1)
        texts = [t for _, t in ex]
        if not texts:
            logger.error("No texts found in examples")
            sys.exit(1)
        tok = ClosedVocabTokenizer.from_texts(texts)
        rows = [{"token": t, "description": ""} for t in tok.vocab]
        write_jsonl(args.out, rows)
        logger.info(f"Wrote {len(rows)} tokens to {args.out}")
        print(f"[build-token-descriptions] wrote {len(rows)} tokens to {args.out}")
    except Exception as e:
        logger.error(f"Error building token descriptions: {e}", exc_info=True)
        sys.exit(1)


def cmd_train(args):
    """Train reasoner model."""
    try:
        logger.info(f"Loading data from {args.data}")
        ex = load_examples(args.data)
        if not ex:
            logger.error(f"No examples found in {args.data}")
            sys.exit(1)
        texts = [t for _, t in ex]
        if not texts:
            logger.error("No texts found in examples")
            sys.exit(1)
        
        token_desc = {}
        if args.token_desc:
            logger.info(f"Loading token descriptions from {args.token_desc}")
            token_desc = load_token_descriptions(args.token_desc)
        
        logger.info(f"Training reasoner: {len(texts)} texts")
        art, _, _ = train_reasoner(
            texts=texts,
            token_desc=token_desc,
            dim=args.dim,
            window=args.window,
            desc_alpha=args.desc_alpha,
            max_vocab_size=args.max_vocab_size,
            bigram_top_k=getattr(args, 'bigram_top_k', 150),
            seed=getattr(args, 'seed', None),
        )
        
        logger.info(f"Saving artifacts to {args.artifacts}")
        art.save(args.artifacts)
        logger.info(f"Training complete: vocab={len(art.vocab)} dim={art.vectors.shape[1]}")
        print(f"[train] vocab={len(art.vocab)} dim={art.vectors.shape[1]} saved to {args.artifacts}")
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        sys.exit(1)


def cmd_train_dual(args):
    """Train dual model (reasoner + selector)."""
    try:
        logger.info(f"Loading data from {args.data}")
        ex = load_examples(args.data)
        if not ex:
            logger.error(f"No examples found in {args.data}")
            sys.exit(1)
        texts = [t for _, t in ex]
        if not texts:
            logger.error("No texts found in examples")
            sys.exit(1)
        
        token_desc = {}
        if args.token_desc:
            logger.info(f"Loading token descriptions from {args.token_desc}")
            token_desc = load_token_descriptions(args.token_desc)
        
        logger.info(f"Training dual model: {len(texts)} texts")
        reasoner, selector = train_dual(
            texts=texts,
            token_desc=token_desc,
            dim=args.dim,
            window=args.window,
            desc_alpha=args.desc_alpha,
            selector_smooth=args.selector_smooth,
            selector_max_per_context=args.selector_max_per_context,
            max_vocab_size=args.max_vocab_size,
            bigram_top_k=getattr(args, 'bigram_top_k', 150),
            seed=getattr(args, 'seed', None),
        )
        
        logger.info(f"Saving artifacts to {args.artifacts}")
        reasoner.save(args.artifacts, selector=selector)  # Save both in model.npz
        logger.info(f"Training complete: vocab={len(reasoner.vocab)} dim={reasoner.vectors.shape[1]}")
        print(f"[train-dual] reasoner vocab={len(reasoner.vocab)} dim={reasoner.vectors.shape[1]} saved to {args.artifacts}")
        print(f"[train-dual] selector contexts={len(selector.trigram_logp)} (trigram) + {len(selector.bigram_logp)} (bigram) saved to {args.artifacts}")
    except Exception as e:
        logger.error(f"Error training dual model: {e}", exc_info=True)
        sys.exit(1)


def cmd_generate(args):
    """Generate text."""
    try:
        logger.info(f"Loading model from {args.artifacts}")
        art = ReasonerArtifacts.load(args.artifacts)
        logger.info(f"Generating text: prompt='{args.prompt[:50]}...', max_tokens={args.max_new_tokens}")
        
        out = generate(
            art=art,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=getattr(args, 'top_p', None),
            explain=bool(args.explain),
            block_dataset_meta=bool(args.block_dataset_meta),
            repeat_window=args.repeat_window,
            repetition_penalty=args.repetition_penalty,
            semantic_repeat_window=args.semantic_repeat_window,
            semantic_repeat_threshold=args.semantic_repeat_threshold,
            semantic_repeat_penalty=args.semantic_repeat_penalty,
            context_window=args.context_window,
            seed=getattr(args, 'seed', None),
        )
        
        print(out["text"])
        if args.explain:
            print("\n--- EXPLANATION (top candidates per step) ---")
            print(json.dumps(out["explain"], ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Error generating text: {e}", exc_info=True)
        sys.exit(1)


def cmd_generate_dual(args):
    """Generate text using dual model."""
    try:
        logger.info(f"Loading models from {args.artifacts}")
        reasoner = ReasonerArtifacts.load(args.artifacts)
        selector = SelectorArtifacts.load(args.artifacts)
        logger.info(f"Generating text: prompt='{args.prompt[:50]}...', max_tokens={args.max_new_tokens}")
        
        out = generate_dual(
            reasoner=reasoner,
            selector=selector,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            selector_top_k=args.selector_top_k,
            alpha_selector=args.alpha_selector,
            beta_reasoner=args.beta_reasoner,
            eta_trust=args.eta_trust,
            eta_bias=args.eta_bias,
            trust_clip=args.trust_clip,
            w_semantic=args.w_semantic,
            w_anchor=args.w_anchor,
            w_contra=args.w_contra,
            w_repeat=args.w_repeat,
            repeat_window=args.repeat_window,
            block_dataset_meta=bool(args.block_dataset_meta),
            explain=bool(args.explain),
            context_window=args.context_window,
            seed=getattr(args, 'seed', None),
        )
        
        print(out["text"])
        if args.explain:
            print("\n--- DEBATE TRACE (top continuations per step) ---")
            print(json.dumps(out.get("explain", []), ensure_ascii=False, indent=2))
            print("\n--- ONLINE STATE ---")
            print(json.dumps(out.get("online", {}), ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Error generating text: {e}", exc_info=True)
        sys.exit(1)


def cmd_vocab(args):
    """Show vocabulary."""
    ex = load_examples(args.data)
    texts = [t for _, t in ex]
    tok = ClosedVocabTokenizer.from_texts(texts)
    print(f"vocab size: {len(tok.vocab)}")
    print(tok.vocab)


def main():
    """Main entry point."""
    p = argparse.ArgumentParser(description="Explainable Token-Reasoning Engine")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    sub = p.add_subparsers(dest="cmd", required=True)
    
    # build-token-descriptions
    p_bt = sub.add_parser("build-token-descriptions")
    p_bt.add_argument("--data", required=True)
    p_bt.add_argument("--out", required=True)
    p_bt.set_defaults(func=cmd_build_token_desc)
    
    # train
    p_tr = sub.add_parser("train")
    p_tr.add_argument("--data", required=True)
    p_tr.add_argument("--token_desc", required=False, default="")
    p_tr.add_argument("--artifacts", required=True)
    p_tr.add_argument("--dim", type=int, default=20, help="Vector dimension (default: 20 for quality)")
    p_tr.add_argument("--window", type=int, default=2)
    p_tr.add_argument("--desc_alpha", type=float, default=0.35)
    p_tr.add_argument("--max_vocab_size", type=int, default=50000)
    p_tr.add_argument("--bigram_top_k", type=int, default=200, help="Top-K transitions per token for sparse bigram (default: 200)")
    p_tr.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p_tr.set_defaults(func=cmd_train)
    
    # train-dual
    p_td = sub.add_parser("train-dual")
    p_td.add_argument("--data", required=True)
    p_td.add_argument("--token_desc", required=False, default="")
    p_td.add_argument("--artifacts", required=True)
    p_td.add_argument("--dim", type=int, default=20, help="Vector dimension (default: 20 for quality)")
    p_td.add_argument("--window", type=int, default=2)
    p_td.add_argument("--desc_alpha", type=float, default=0.35)
    p_td.add_argument("--selector_smooth", type=float, default=0.5)
    p_td.add_argument("--selector_max_per_context", type=int, default=256)
    p_td.add_argument("--max_vocab_size", type=int, default=50000)
    p_td.add_argument("--bigram_top_k", type=int, default=200, help="Top-K transitions per token for sparse bigram (default: 200)")
    p_td.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p_td.set_defaults(func=cmd_train_dual)
    
    # generate
    p_ge = sub.add_parser("generate")
    p_ge.add_argument("--artifacts", required=True)
    p_ge.add_argument("--prompt", required=True)
    p_ge.add_argument("--max_new_tokens", type=int, default=60)
    p_ge.add_argument("--temperature", type=float, default=0.75)
    p_ge.add_argument("--top_k", type=int, default=6)
    p_ge.add_argument("--top_p", type=float, default=None)
    p_ge.add_argument("--explain", type=int, default=0)
    p_ge.add_argument("--block_dataset_meta", type=int, default=1)
    p_ge.add_argument("--repeat_window", type=int, default=6)
    p_ge.add_argument("--repetition_penalty", type=float, default=1.5)
    p_ge.add_argument("--semantic_repeat_window", type=int, default=2)
    p_ge.add_argument("--semantic_repeat_threshold", type=float, default=0.6)
    p_ge.add_argument("--semantic_repeat_penalty", type=float, default=0.85)
    p_ge.add_argument("--context_window", type=int, default=40)
    p_ge.add_argument("--seed", type=int, default=None)
    p_ge.set_defaults(func=cmd_generate)
    
    # generate-dual
    p_gd = sub.add_parser("generate-dual")
    p_gd.add_argument("--artifacts", required=True)
    p_gd.add_argument("--prompt", required=True)
    p_gd.add_argument("--max_new_tokens", type=int, default=60)
    p_gd.add_argument("--temperature", type=float, default=0.85)
    p_gd.add_argument("--selector_top_k", type=int, default=32)
    p_gd.add_argument("--alpha_selector", type=float, default=0.55)
    p_gd.add_argument("--beta_reasoner", type=float, default=0.45)
    p_gd.add_argument("--eta_trust", type=float, default=0.10)
    p_gd.add_argument("--eta_bias", type=float, default=0.08)
    p_gd.add_argument("--trust_clip", type=float, default=2.0)
    p_gd.add_argument("--w_semantic", type=float, default=1.0)
    p_gd.add_argument("--w_anchor", type=float, default=0.55)
    p_gd.add_argument("--w_contra", type=float, default=1.25)
    p_gd.add_argument("--w_repeat", type=float, default=0.85)
    p_gd.add_argument("--repeat_window", type=int, default=6)
    p_gd.add_argument("--context_window", type=int, default=20, help="Context window size (increased for better coherence)")
    p_gd.add_argument("--block_dataset_meta", type=int, default=1)
    p_gd.add_argument("--explain", type=int, default=0)
    p_gd.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p_gd.set_defaults(func=cmd_generate_dual)
    
    # vocab
    p_v = sub.add_parser("vocab")
    p_v.add_argument("--data", required=True)
    p_v.set_defaults(func=cmd_vocab)
    
    args = p.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

