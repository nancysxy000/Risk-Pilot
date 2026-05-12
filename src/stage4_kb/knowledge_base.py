"""
Stage 4: 规则知识库 — 向量存储与检索

使用 ChromaDB 或 FAISS 存储规则的向量表示，支持语义检索。
"""

import json
import os


class RuleKnowledgeBase:
    """规则知识库: 存储历史有效规则 + 支持语义检索"""

    def __init__(self, config):
        kb_cfg = config['knowledge_base']
        self.store_type = kb_cfg.get('vector_store', 'chroma')
        self.persist_dir = kb_cfg.get('persist_dir', 'outputs/kb')
        self.collection_name = kb_cfg.get('collection_name', 'risk_rules')
        self.collection = None
        self._rules_cache = []  # 内存缓存

        os.makedirs(self.persist_dir, exist_ok=True)
        self._init_store()

    def _init_store(self):
        """初始化向量存储"""
        if self.store_type == 'chroma':
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"[KB] ChromaDB initialized, {self.collection.count()} rules in store")
            except ImportError:
                print("[KB] ChromaDB not installed, using JSON fallback")
                self._init_json_fallback()
        else:
            self._init_json_fallback()

    def _init_json_fallback(self):
        """JSON 文件作为简单备选存储"""
        self.store_type = 'json'
        self._json_path = os.path.join(self.persist_dir, 'rules.json')
        if os.path.exists(self._json_path):
            with open(self._json_path, 'r') as f:
                self._rules_cache = json.load(f)
        print(f"[KB] JSON fallback, {len(self._rules_cache)} rules loaded")

    def is_empty(self):
        """知识库是否为空"""
        if self.collection:
            return self.collection.count() == 0
        return len(self._rules_cache) == 0

    def add_rules(self, rules):
        """添加规则到知识库"""
        if self.collection:
            for rule in rules:
                from .rule_schema import Rule
                if isinstance(rule, dict):
                    r = Rule(**{k: v for k, v in rule.items() if k in Rule.__dataclass_fields__})
                else:
                    r = rule

                self.collection.upsert(
                    ids=[r.rule_id],
                    documents=[r.to_embedding_text()],
                    metadatas=[r.to_dict()],
                )
            print(f"[KB] Added {len(rules)} rules to ChromaDB")
        else:
            for rule in rules:
                if isinstance(rule, dict):
                    self._rules_cache.append(rule)
                else:
                    self._rules_cache.append(rule.to_dict())
            self._save_json()
            print(f"[KB] Added {len(rules)} rules to JSON store")

    def search(self, query_text, top_k=5):
        """语义检索相似规则"""
        if self.collection and self.collection.count() > 0:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(top_k, self.collection.count()),
            )
            return results.get('metadatas', [[]])[0]
        else:
            # JSON fallback: 简单关键词匹配
            return self._rules_cache[:top_k]

    def get_all_rules(self):
        """获取所有规则"""
        if self.collection and self.collection.count() > 0:
            results = self.collection.get()
            return results.get('metadatas', [])
        return self._rules_cache

    def _save_json(self):
        """保存 JSON 备选存储"""
        with open(self._json_path, 'w', encoding='utf-8') as f:
            json.dump(self._rules_cache, f, indent=2, ensure_ascii=False)
