# MidRS

python

from app.models import db

db.drop_all()

db.create_all()

from app.models import Users

user1 = Users(field=...)

db.session.add(user1)

db.session.commit()
