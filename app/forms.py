from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, TextAreaField, RadioField, IntegerField
from wtforms.validators import Length, Email, DataRequired, EqualTo, ValidationError, NumberRange

from app.models import Users, Types, Algorithms

class RegisterForm(FlaskForm):
    def validate_user_name(self, user_name_to_check):
        user = Users.query.filter_by(user_name=user_name_to_check.data).first()
        if user:
            raise ValidationError('Tên đăng nhập đã được sử dụng!')

    def validate_email_address(self, email_address_to_check):
        email_address = Users.query.filter_by(
            email_address=email_address_to_check.data).first()
        if email_address:
            raise ValidationError('Email đã được sử dụng!')

    user_name = StringField(label='Tên đăng nhập:', validators=[
                            Length(min=4, max=30), DataRequired()])
    email_address = StringField(label='Email:', validators=[
                                Email(), DataRequired()])
    password = PasswordField(label='Mật khẩu:', validators=[
                             Length(min=6), DataRequired()])
    confirm_password = PasswordField(label='Xác nhận mật khẩu:', validators=[EqualTo(
        'password', message='Xác nhận mật khẩu không trùng khớp!'), DataRequired()])
    submit = SubmitField(label='Tạo tài khoản')


class LoginFrom(FlaskForm):
    user_name = StringField(label='Tên đăng nhập:',
                            validators=[DataRequired()])
    password = PasswordField(label='Mật khẩu:', validators=[DataRequired()])
    submit = SubmitField(label='Đăng nhập')


class AlgorithmForm(FlaskForm):
    name = StringField(label='Tên:', validators=[DataRequired()])
    description = TextAreaField(label='Mô tả:', validators=[DataRequired()])
    link = StringField(label='Đường dẫn thông tin bổ sung:', validators=[DataRequired()])
    submit = SubmitField(label='Thêm mới')

class TypeForm(FlaskForm):
    name = StringField(label='Tên:', validators=[DataRequired()])
    submit = SubmitField(label='Thêm mới')

choices = [
    (0, 'Tối ưu'),
    (1, 'Phổ biến'),
    (2, 'Lọc cộng tác (Người dùng - Độ tương đồng Cosine)'),
    (3, 'Lọc cộng tác (Người dùng - Độ tương đồng Pearson)'),
    (4, 'Lọc cộng tác (Người dùng - Độ tương đồng Jaccard)'),
    (5, 'Lọc cộng tác (Chỉ  mục - Độ tương đồng Cosine)'),
    (6, 'Lọc cộng tác (Chỉ  mục - Độ tương đồng Pearson)'),
    (7, 'Lọc cộng tác (Chỉ  mục - Độ tương đồng Jaccard)'),
    (8, 'Ma trận phân rã'),

    # (8, 'Ma trận phân rã (Người dùng)'),
#     (9, 'Ma trận phân rã (Chỉ mục)'),
]


types = [
    (0, 'Phim'),
    (1, 'Sách'),
    (2, 'Khác'),

]


class UploadDataForm(FlaskForm):
    file = FileField(label='Tải lên tập tin dữ liệu:', validators=[DataRequired()])
    select = RadioField(label='Thuật toán:', validators=[DataRequired()])
    # type = RadioField(label='Loại dữ liệu:', choices=types, validators=[DataRequired()])
    type = RadioField(label='Loại dữ liệu:', validators=[DataRequired()])

    amount = IntegerField(label='Số lượng gợi ý tối đa:', validators=[NumberRange(min=1, message='Số lượng không thể âm!'), DataRequired()])
    submit = SubmitField(label='Bắt đầu tính toán')

    def __init__(self, *args, **kwargs):
        super(UploadDataForm, self).__init__(*args, **kwargs)
        self.type.choices = [(type.id, type.name) 
                                        for type in Types.query.all()]
        self.select.choices = [(algorithm.id, algorithm.name) 
                                        for algorithm in Algorithms.query.all()]

class UploadTemplateForm(FlaskForm):
    file = FileField(label='Tải lên tập tin mẫu:', validators=[DataRequired()])
    submit = SubmitField(label='Tải lên tập tin mẫu')

class ReHandleForm(FlaskForm):
    select = RadioField(label='Thuật toán:', validators=[DataRequired()])
    submit = SubmitField(label='Bắt đầu tính toán')

    def __init__(self, *args, **kwargs):
        super(ReHandleForm, self).__init__(*args, **kwargs)
        self.select.choices = [(algorithm.id, algorithm.name) 
                                        for algorithm in Algorithms.query.all()]
