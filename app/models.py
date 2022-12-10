from app import db, bcrypt, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

class Users(db.Model,UserMixin):
    id = db.Column(db.Integer(), primary_key=True)
    user_name = db.Column(db.String(length=30), nullable=False, unique=True)
    password_hash = db.Column(db.String(length=60), nullable=False)
    email_address = db.Column(db.String(length=50), nullable=False, unique=True)
    is_admin = db.Column(db.Boolean(), default=False, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())
    files = db.relationship('Files', backref='files_user', lazy=True)

    @property
    def password(self):
        return self.password

    @password.setter
    def password(self, password_form_register):
        self.password_hash = bcrypt.generate_password_hash(password_form_register).decode('utf-8')

    def check_password_correction(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class Files(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(length=50), nullable=False)
    storage_path = db.Column(db.String(length=100), nullable=False)
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id'), nullable=False)
    type_id = db.Column(db.Integer(), db.ForeignKey('types.id'), nullable=True)
    other = db.Column(db.String(length=50),nullable=True)
    amount = db.Column(db.Integer(), default=10, nullable=True)
    is_template = db.Column(db.Boolean(), default=True, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())
    results = db.relationship('Results', backref='results_file', lazy=True)

class Algorithms(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(length=50), nullable=False)
    description = db.Column(db.String(1000), nullable=False)
    link = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())
    results = db.relationship('Results', backref='results_algorithm', lazy=True)
    points = db.relationship('Points', backref='points_algorithm', lazy=True)


class Types(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(length=50), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())
    files = db.relationship('Files', backref='files_type', lazy=True)

class Results(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(length=50), nullable=False)
    storage_path = db.Column(db.String(length=100), nullable=False)
    file_upload_id = db.Column(db.Integer(), db.ForeignKey('files.id'), nullable=False)
    algorithm_id = db.Column(db.Integer(), db.ForeignKey('algorithms.id'), nullable=False)
    knn = db.Column(db.Integer(), nullable=True)
    downloads = db.Column(db.Integer(), default=0, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())
    points = db.relationship('Points', backref='points_result', lazy=True)

class Points(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    result_id = db.Column(db.Integer(), db.ForeignKey('results.id'), nullable=False)
    algorithm_id = db.Column(db.Integer(), db.ForeignKey('algorithms.id'), nullable=False)
    rmse = db.Column(db.Float(), nullable=True)
    mae = db.Column(db.Float(), nullable=True)
    precision = db.Column(db.Float(), nullable=True)
    recall = db.Column(db.Float(), nullable=True)
    f1 = db.Column(db.Float(), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())

# class User(db.Model, UserMixin):
#     id = db.Column(db.Integer(), primary_key=True)
#     username = db.Column(db.String(length=30), nullable=False, unique=True)
#     email_address = db.Column(db.String(length=50),
#                               nullable=False, unique=True)
#     password_hash = db.Column(db.String(length=60), nullable=False)
#     budget = db.Column(db.Integer(), nullable=False, default=1000)
#     items = db.relationship('Item', backref='owned_user', lazy=True)

#     @property
#     def password(self):
#         return self.password

#     @password.setter
#     def password(self, plain_text_password):
#         self.password_hash = bcrypt.generate_password_hash(
#             plain_text_password).decode('utf-8')

#     def check_password_correction(self, attempted_password):
#         return bcrypt.check_password_hash(self.password_hash, attempted_password)


# class Item(db.Model):
#     id = db.Column(db.Integer(), primary_key=True)
#     name = db.Column(db.String(length=30), nullable=False, unique=True)
#     price = db.Column(db.Integer(), nullable=False)
#     barcode = db.Column(db.String(length=12), nullable=False, unique=True)
#     description = db.Column(db.String(length=1024),
#                             nullable=False, unique=True)
#     owner = db.Column(db.Integer(), db.ForeignKey('user.id'))


# i = Item.query....
# i.owner => id of user
# i.owned_user => instance of user
