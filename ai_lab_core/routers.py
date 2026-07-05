class AiLabRouter:
    """
    Routes all models in ai_lab.* apps to the ai_lab database.
    Everything else (casa's existing models) goes to default.
    """

    ai_lab_app_prefix = 'ai_lab'

    def _is_ai_lab(self, model):
        return model._meta.app_label.startswith(self.ai_lab_app_prefix)

    def db_for_read(self, model, **hints):
        return 'ai_lab' if self._is_ai_lab(model) else None

    def db_for_write(self, model, **hints):
        return 'ai_lab' if self._is_ai_lab(model) else None

    def allow_relation(self, obj1, obj2, **hints):
        # Allow relations only within the same database
        db1 = 'ai_lab' if self._is_ai_lab(type(obj1)) else 'default'
        db2 = 'ai_lab' if self._is_ai_lab(type(obj2)) else 'default'
        return db1 == db2

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        # ai_lab apps migrate only to ai_lab DB; everything else only to default
        if app_label.startswith(self.ai_lab_app_prefix):
            return db == 'ai_lab'
        return db == 'default'