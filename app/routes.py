import csv
from datetime import datetime, timedelta
import io
import os
import pathlib
from wtforms import Label
from algorithms.collaborative_filtering import CollaborativeFiltering
from flask import flash, make_response, redirect, render_template, send_file, url_for, request,  jsonify
from algorithms.matrix_factorization import MatrixFactorizattion
from algorithms.popular import Popular
from app import app, db
from app.forms import LoginFrom, RegisterForm, UploadDataForm, UploadTemplateForm, TypeForm, AlgorithmForm, ReHandleForm
from app.models import Files, Users, Types, Algorithms, Results, Points
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from algorithms.constants import *
from sklearn.model_selection import train_test_split
from sqlalchemy import desc
import json


@login_required
def check_is_admin():
    if current_user.is_admin == 0:
        redirect(url_for('home_page'))


@app.route("/")
@login_required
def home_page():
    algorithms = Algorithms.query.all()
    return render_template("home_page.html", algorithms = algorithms)


@app.route("/users")
@login_required
def user_page():
    users = Users.query.all()
    return render_template("user_page.html", users=users)

@app.route("/users/<int:id>")
@login_required
def activity_user_page(id):
    files = Files.query.filter_by(user_id=id).all()
    algorithms = Algorithms.query.all()
    results = Results.query.all()
    return render_template("activity_user_page.html", files=files, algorithms = algorithms,results=results)

def get_data(data):
    ratings = pd.read_csv(data, sep="[;, \t]",  encoding='latin-1',
                          on_bad_lines='skip', engine='python', usecols=[0, 1, 2], header=None)
    ratings.dropna()
    Y_data = ratings.to_numpy()
    return Y_data


def check_binary(array):
    return np.array_equal(array, array.astype(bool))


@app.route("/files", methods=['GET', 'POST'])
@login_required
def file_page():
    form = UploadDataForm()
    files = Files.query.filter_by(user_id=current_user.id, is_template=False).order_by(desc(Files.created_at))
    types = Types.query.all()

    filename = ''
    filename_result = ''
    if form.file.data:
        filename = secure_filename(
            str(datetime.now())+'-'+form.file.data.filename)
        filename_result = secure_filename(
            str(datetime.now() + timedelta(seconds=10))+'-'+form.file.data.filename)

    if form.validate_on_submit():
        storage_path = 'uploads\\' + filename
        storage_path_result = 'uploads\\' + filename_result
        other = request.form['other']
        knn = request.form['knn']
        data = get_data(form.file.data)
        

        if check_binary(data[:,2]):
            if form.select.data == '5' or form.select.data == '8' or form.select.data == '1' or form.select.data == '2':

                with open(storage_path, 'w', newline="") as f:
                    write = csv.writer(f, delimiter=" ")
                    write.writerows(data)

                file_upload = Files(name=form.file.data.filename, storage_path=storage_path,
                                    user_id=current_user.id, is_template=0, type_id=form.type.data, other=other, amount=form.amount.data)
                db.session.add(file_upload)
                db.session.commit()

                amount = int(str(form.amount.data))

                Y_train, Y_test = train_test_split(data)

                if form.select.data == '5':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=JACCARD, based=USER)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    precision, recall = rs.evaluate()
                    f1 = 0
                    if precision != 0 or recall != 0:
                        f1 = 2*precision*recall/(precision+recall)
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, precision=precision, recall=recall, f1=f1)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                    return redirect(url_for('file_page')) 
                elif form.select.data == '8':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=JACCARD, based=ITEM)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    precision, recall = rs.evaluate()
                    f1 = 0
                    if precision != 0 or recall != 0:
                        f1 = 2*precision*recall/(precision+recall)
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, precision=precision, recall=recall, f1=f1)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                    return redirect(url_for('file_page')) 
                elif form.select.data == '1':
                    #build model
                    rs_cf_jaccard_user = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=JACCARD, based=USER)
                    rs_cf_jaccard_user.fit()

                    rs_cf_jaccard_item = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=JACCARD, based=ITEM)
                    rs_cf_jaccard_item.fit()

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()  

                    #evaluation
                    eva_jaccard_user = rs_cf_jaccard_user.evaluate()
                    f1_jaccard_user = 0 
                    if eva_jaccard_user[0] != 0 or eva_jaccard_user[1] != 0:
                        f1_jaccard_user = 2*eva_jaccard_user[0]*eva_jaccard_user[1]/(eva_jaccard_user[0]+eva_jaccard_user[1])       
                    point_jaccard_user = Points(result_id=result.id,
                                algorithm_id=5, precision=eva_jaccard_user[0], recall=eva_jaccard_user[1], f1=f1_jaccard_user)
                    db.session.add(point_jaccard_user)

                    eva_jaccard_item = rs_cf_jaccard_item.evaluate()
                    f1_jaccard_item = 0 
                    if eva_jaccard_item[0] != 0 or eva_jaccard_item[1] != 0:
                        f1_jaccard_item = 2*eva_jaccard_item[0]*eva_jaccard_item[1]/(eva_jaccard_item[0]+eva_jaccard_item[1])       
                    point_jaccard_item = Points(result_id=result.id,
                                algorithm_id=8, precision=eva_jaccard_item[0], recall=eva_jaccard_item[1], f1=f1_jaccard_item)
                    db.session.add(point_jaccard_item)
                    db.session.commit()

                    #different
                    if f1_jaccard_user>f1_jaccard_item:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_jaccard_user.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                        return redirect(url_for('file_page'))  
                    else:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_jaccard_item.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                        return redirect(url_for('file_page'))  
            else:
                flash('Giải thuật không hỗ trợ trên dữ liệu nhị phân! Xin vui lòng thử lại!', category='warning')    
        else:  
            if form.select.data != '5' and form.select.data != '8':

                with open(storage_path, 'w', newline="") as f:
                    write = csv.writer(f, delimiter=" ")
                    write.writerows(data)

                file_upload = Files(name=form.file.data.filename, storage_path=storage_path,
                                    user_id=current_user.id, is_template=0, type_id=form.type.data, other=other, amount=form.amount.data)
                db.session.add(file_upload)
                db.session.commit()

                amount = int(str(form.amount.data))
                Y_train, Y_test = train_test_split(data)
                if form.select.data == '2':
                    rs = Popular(Y_data=Y_train, Y_test=Y_test, k=amount)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result,
                                    file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                    return redirect(url_for('file_page'))
                elif form.select.data == '3':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=COSINE, based=USER)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                    return redirect(url_for('file_page'))  
                elif form.select.data == '4':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=PEARSON, based=USER)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                    return redirect(url_for('file_page')) 
                elif form.select.data == '6':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=COSINE, based=ITEM)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                    return redirect(url_for('file_page')) 
                elif form.select.data == '7':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=PEARSON, based=ITEM)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                    return redirect(url_for('file_page'))
                elif form.select.data == '9':
                    rs = MatrixFactorizattion(Y_data = Y_train, Y_test=Y_test, amount = amount, latent=10, regularization=.1, eta=0.70)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result,
                                    file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                    return redirect(url_for('file_page'))
                else:
                    # build model
                    rs_popular = Popular(Y_data=Y_train, Y_test=Y_test, k=amount)
                    rs_popular.fit()

                    rs_cf_cosine_user = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=COSINE, based=USER)
                    rs_cf_cosine_user.fit()

                    rs_cf_pearson_user = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=PEARSON, based=USER)
                    rs_cf_pearson_user.fit()

                    rs_cf_cosine_item = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=COSINE, based=ITEM)
                    rs_cf_cosine_item.fit()

                    rs_cf_pearson_item = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=PEARSON, based=ITEM)
                    rs_cf_pearson_item.fit()

                    rs_mf = MatrixFactorizattion(Y_data = Y_train, Y_test=Y_test, amount = amount, latent=10, regularization=.1, eta=0.70)
                    rs_mf.fit()

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                        file_upload_id=file_upload.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    # evaluation
                    eva_popular = rs_popular.evaluate()
                    point_popular = Points(result_id=result.id,
                                    algorithm_id=2, rmse=eva_popular[0], mae=eva_popular[1])
                    db.session.add(point_popular)

                    eva_cosine_user = rs_cf_cosine_user.evaluate()
                    point_cosine_user = Points(result_id=result.id,
                                    algorithm_id=3, rmse=eva_cosine_user[0], mae=eva_cosine_user[1])
                    db.session.add(point_cosine_user)

                    eva_pearson_user = rs_cf_pearson_user.evaluate()
                    point_pearson_user = Points(result_id=result.id,
                                    algorithm_id=4, rmse=eva_pearson_user[0], mae=eva_pearson_user[1])
                    db.session.add(point_pearson_user)

                    eva_cosine_item = rs_cf_cosine_item.evaluate()
                    point_cosine_item = Points(result_id=result.id,
                                    algorithm_id=6, rmse=eva_cosine_item[0], mae=eva_cosine_item[1])
                    db.session.add(point_cosine_item)

                    eva_pearson_item = rs_cf_pearson_item.evaluate()
                    point_pearson_item = Points(result_id=result.id,
                                    algorithm_id=7, rmse=eva_pearson_item[0], mae=eva_pearson_item[1])
                    db.session.add(point_pearson_item)

                    eva_mf = rs_mf.evaluate()
                    point_mf = Points(result_id=result.id,
                                    algorithm_id=9, rmse=eva_mf[0], mae=eva_mf[1])
                    db.session.add(point_mf)
                    db.session.commit()
                    # different
                    array_eva = [eva_popular[0], eva_cosine_user[0], eva_pearson_user[0], eva_cosine_item[0], eva_pearson_item[0], eva_mf[0]]
                    index_min = array_eva.index(min(array_eva))

                    if index_min == 0:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_popular.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                        return redirect(url_for('file_page'))
                    elif index_min == 1:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_cosine_user.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                        return redirect(url_for('file_page'))
                    elif index_min == 2:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_pearson_user.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                        return redirect(url_for('file_page'))
                    elif index_min == 3:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_cosine_item.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                        return redirect(url_for('file_page'))
                    elif index_min == 4:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_pearson_item.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                        return redirect(url_for('file_page'))
                    else:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_mf.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                        return redirect(url_for('file_page'))
            else:
                flash('Giải thuật không hỗ trợ trên dữ liệu phi nhị phân! Xin vui lòng thử lại!', category='warning')  
    
    if form.errors != {}:
        # for error in form.errors.values():
        # flash(error[0], category='danger')
        flash('Lỗi! Xin vui lòng thử lại!', category='danger')
    return render_template("file_page.html", form=form, files=files, types=types)


@app.route("/download/<int:id>", methods=['GET', 'POST'])
@login_required
def download(id):
    result = Results.query.filter_by(id=id).first()
    result.downloads += 1
    db.session.commit()
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    path = os.path.join(parent_path, result.storage_path)
    return send_file(path, as_attachment=True)

# @app.route("/show/<int:id>", methods=['GET', 'POST'])
# @login_required
# def show(id):
#     file = Files.query.filter_by(id=id).first()
#     parent_path = pathlib.Path(__file__).parent.parent.resolve()
#     path = os.path.join(parent_path, file.link)
#     data = []
#     max = 1
#     with open(path) as f:
#         d = csv.reader(f)
#         for row in d:
#             row = row[0].split(" ")
#             if max < len(row): max = len(row)
#             data.append(list(row[1:]))
#     return render_template("show.html", data=data, max=max)


@app.route("/results/<int:id>/", methods=['GET', 'POST'])
@login_required
def result_page(id):
    results = Results.query.filter_by(file_upload_id=id).all()
    algorithms = Algorithms.query.all()
    form = ReHandleForm()
    if form.validate_on_submit():
        file = Files.query.filter_by(id=id).first()
        filename_result = secure_filename(
            str(datetime.now())+'-'+file.name)
        knn = request.form['knn']

        data = get_data(file.storage_path)
        if check_binary(data[:,2]):
            if form.select.data == '5' or form.select.data == '8' or form.select.data == '1' or form.select.data == '2':
                storage_path_result = 'uploads\\' + filename_result
                amount = int(str(file.amount))
                Y_train, Y_test = train_test_split(data)

                if form.select.data == '5':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=JACCARD, based=USER)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    precision, recall = rs.evaluate()
                    f1 = 0
                    if precision != 0 or recall != 0:
                        f1 = 2*precision*recall/(precision+recall)
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, precision=precision, recall=recall, f1=f1)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                elif form.select.data == '8':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=JACCARD, based=ITEM)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    precision, recall = rs.evaluate()
                    f1 = 0
                    if precision != 0 or recall != 0:
                        f1 = 2*precision*recall/(precision+recall)
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, precision=precision, recall=recall, f1=f1)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                elif form.select.data == '1':
                    #build model
                    rs_cf_jaccard_user = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=JACCARD, based=USER)
                    rs_cf_jaccard_user.fit()

                    rs_cf_jaccard_item = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=JACCARD, based=ITEM)
                    rs_cf_jaccard_item.fit()

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()  

                    #evaluation
                    eva_jaccard_user = rs_cf_jaccard_user.evaluate()
                    f1_jaccard_user = 0 
                    if eva_jaccard_user[0] != 0 or eva_jaccard_user[1] != 0:
                        f1_jaccard_user = 2*eva_jaccard_user[0]*eva_jaccard_user[1]/(eva_jaccard_user[0]+eva_jaccard_user[1])       
                    point_jaccard_user = Points(result_id=result.id,
                                algorithm_id=5, precision=eva_jaccard_user[0], recall=eva_jaccard_user[1], f1=f1_jaccard_user)
                    db.session.add(point_jaccard_user)

                    eva_jaccard_item = rs_cf_jaccard_item.evaluate()
                    f1_jaccard_item = 0 
                    if eva_jaccard_item[0] != 0 or eva_jaccard_item[1] != 0:
                        f1_jaccard_item = 2*eva_jaccard_item[0]*eva_jaccard_item[1]/(eva_jaccard_item[0]+eva_jaccard_item[1])       
                    point_jaccard_item = Points(result_id=result.id,
                                algorithm_id=8, precision=eva_jaccard_item[0], recall=eva_jaccard_item[1], f1=f1_jaccard_item)
                    db.session.add(point_jaccard_item)
                    db.session.commit()

                    #different
                    if f1_jaccard_user>f1_jaccard_item:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_jaccard_user.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                    else:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_jaccard_item.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                return redirect(url_for('result_page',id=id))
            else:
                flash('Giải thuật không hỗ trợ trên dữ liệu nhị phân! Xin vui lòng thử lại!', category='warning')    
        else:  
            if form.select.data != '5' and form.select.data != '8':
                storage_path_result = 'uploads\\' + filename_result
                amount = int(str(file.amount))
                Y_train, Y_test = train_test_split(data)
                if form.select.data == '2':
                    rs = Popular(Y_data=Y_train, Y_test=Y_test, k=amount)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result,
                                    file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                elif form.select.data == '3':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=COSINE, based=USER)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                elif form.select.data == '4':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=PEARSON, based=USER)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                elif form.select.data == '6':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=COSINE, based=ITEM)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                elif form.select.data == '7':
                    rs = CollaborativeFiltering(Y_data = Y_train, Y_test=Y_test, k=int(str(knn)), amount = amount, similarity_based=PEARSON, based=ITEM)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                    file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                elif form.select.data == '9':
                    rs = MatrixFactorizattion(Y_data = Y_train, Y_test=Y_test, amount = amount, latent=10, regularization=.1, eta=0.70)
                    rs.fit()
                    with open(storage_path_result, 'w', newline="") as f:
                        write = csv.writer(f, delimiter=" ")
                        write.writerows(rs.recommendation_result())

                    result = Results(name=filename_result, storage_path=storage_path_result,
                                    file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    RMSE, MAE = rs.evaluate()
                    point = Points(result_id=result.id,
                                algorithm_id=form.select.data, rmse=RMSE, mae=MAE)
                    db.session.add(point)
                    db.session.commit()
                    flash('Tải tập tin lên thành công!', category='success')
                else:
                    # build model
                    rs_popular = Popular(Y_data=Y_train, Y_test=Y_test, k=amount)
                    rs_popular.fit()

                    rs_cf_cosine_user = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=COSINE, based=USER)
                    rs_cf_cosine_user.fit()

                    rs_cf_pearson_user = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=PEARSON, based=USER)
                    rs_cf_pearson_user.fit()

                    rs_cf_cosine_item = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=COSINE, based=ITEM)
                    rs_cf_cosine_item.fit()

                    rs_cf_pearson_item = CollaborativeFiltering(Y_data = Y_train, Y_test= Y_test, k=int(str(knn)), amount = amount, similarity_based=PEARSON, based=ITEM)
                    rs_cf_pearson_item.fit()

                    rs_mf = MatrixFactorizattion(Y_data = Y_train, Y_test=Y_test, amount = amount, latent=10, regularization=.1, eta=0.70)
                    rs_mf.fit()

                    result = Results(name=filename_result, storage_path=storage_path_result, knn=int(str(knn)),
                                        file_upload_id=file.id, algorithm_id=form.select.data)
                    db.session.add(result)
                    db.session.commit()
                    # evaluation
                    eva_popular = rs_popular.evaluate()
                    point_popular = Points(result_id=result.id,
                                    algorithm_id=2, rmse=eva_popular[0], mae=eva_popular[1])
                    db.session.add(point_popular)

                    eva_cosine_user = rs_cf_cosine_user.evaluate()
                    point_cosine_user = Points(result_id=result.id,
                                    algorithm_id=3, rmse=eva_cosine_user[0], mae=eva_cosine_user[1])
                    db.session.add(point_cosine_user)

                    eva_pearson_user = rs_cf_pearson_user.evaluate()
                    point_pearson_user = Points(result_id=result.id,
                                    algorithm_id=4, rmse=eva_pearson_user[0], mae=eva_pearson_user[1])
                    db.session.add(point_pearson_user)

                    eva_cosine_item = rs_cf_cosine_item.evaluate()
                    point_cosine_item = Points(result_id=result.id,
                                    algorithm_id=6, rmse=eva_cosine_item[0], mae=eva_cosine_item[1])
                    db.session.add(point_cosine_item)

                    eva_pearson_item = rs_cf_pearson_item.evaluate()
                    point_pearson_item = Points(result_id=result.id,
                                    algorithm_id=7, rmse=eva_pearson_item[0], mae=eva_pearson_item[1])
                    db.session.add(point_pearson_item)

                    eva_mf = rs_mf.evaluate()
                    point_mf = Points(result_id=result.id,
                                    algorithm_id=9, rmse=eva_mf[0], mae=eva_mf[1])
                    db.session.add(point_mf)
                    db.session.commit()
                    # different
                    array_eva = [eva_popular[0], eva_cosine_user[0], eva_pearson_user[0], eva_cosine_item[0], eva_pearson_item[0], eva_mf[0]]
                    index_min = array_eva.index(min(array_eva))

                    if index_min == 0:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_popular.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                    elif index_min == 1:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_cosine_user.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                    elif index_min == 2:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_pearson_user.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                    elif index_min == 3:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_cosine_item.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                    elif index_min == 4:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_cf_pearson_item.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                    else:
                        with open(storage_path_result, 'w', newline="") as f:
                            write = csv.writer(f, delimiter=" ")
                            write.writerows(rs_mf.recommendation_result())
                        flash('Tải tập tin lên thành công!', category='success')
                return redirect(url_for('result_page',id=id))
            else:
                flash('Giải thuật không hỗ trợ trên dữ liệu phi nhị phân! Xin vui lòng thử lại!', category='warning')  
    if form.errors != {}:
        for error in form.errors.values():
            flash(error[0], category='danger')
    return render_template("result_page.html", results=results, algorithms=algorithms, form=form)


@app.route("/chart/<int:id>", methods=['GET', 'POST'])
@login_required
def show_chart(id):
    algorithms_data = []
    points_data = []
    algorithms = Algorithms.query.all()
    for algorithm in algorithms:
        algorithms_data.append((algorithm.id, algorithm.name))
    points = Points.query.filter_by(result_id=id).all()
    for point in points:
        points_data.append((point.algorithm_id, point.rmse, point.mae, point.precision, point.recall, point.f1))
    return jsonify({'algorithms':algorithms_data, 'points': points_data})


@app.route("/views/<int:id>", methods=['GET', 'POST'])
@login_required
def view_page(id):
    result = Results.query.filter_by(id=id).first()
    data = []
    max = 1
    with open(result.storage_path) as f:
        d = csv.reader(f)
        for row in d:
            row = row[0].split(" ")
            if max < len(row):
                max = len(row)
            data.append(list(row[1:]))
    return render_template("view_page.html", data=data, max=max)


@app.route("/files/sample", methods=['GET', 'POST'])
@login_required
def download_sample():
    file = Files.query.filter_by(is_template=1).order_by(desc(Files.created_at)).first()
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    path = os.path.join(parent_path, file.storage_path)
    # data = get_data(path)
    # si = io.StringIO()
    # cw = csv.writer(si)
    # cw.writerows(data)
    # output = make_response(si.getvalue())
    # # si = io.StringIO()
    # # cw = csv.writer(si)
    # # cw.writerows(rs.recommendation_result())
    # # output = make_response(si.getvalue())
    # # result = rs.recommendation_result()
    # # for i in range(len(result[0])):
    # # f.write(str(output))
    # output.headers[
    #     "Content-Disposition"] = f'attachment; filename={file.name.split(".")[0]}_{str(datetime.now())}.csv'
    # output.headers["Content-type"] = "text/csv"
    # return output
    return send_file(path, as_attachment=True)


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user = Users(user_name=form.user_name.data,
                     email_address=form.email_address.data, password=form.password.data)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login_page'))
    if form.errors != {}:
        for error in form.errors.values():
            flash(error[0], category='danger')
    return render_template('register_page.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginFrom()
    if form.validate_on_submit():
        user = Users.query.filter_by(user_name=form.user_name.data).first()
        if user and user.check_password_correction(password=form.password.data):
            login_user(user)
            flash(
                f'Bạn đã đăng nhập thành công, người dùng: {user.user_name}', category='success')
            if current_user.is_admin == 0:
                return redirect(url_for('home_page'))
            return redirect(url_for('admin_page'))
        else:
            flash(f'Đăng nhập thất bại!', category='danger')
    if form.errors != {}:
        for error in form.errors.values():
            flash(error[0], category='danger')
    return render_template('login_page.html', form=form)


@app.route('/logout')
def logout():
    logout_user()
    flash('Bạn đã đăng xuất!', category='info')
    return redirect(url_for('login_page'))


@app.route("/admin")
@login_required
def admin_page():
    check_is_admin()
    count_files = Files.query.filter_by(is_template=0).count()
    result = Results.query.filter_by()
    data = {'template': count_files, 'upload': result.count()}
    data_temp = db.session.query(Algorithms.name,db.func.count(Results.algorithm_id)).outerjoin(Results, Algorithms.id == Results.algorithm_id).group_by(Algorithms.name).order_by(Algorithms.id).all()

    return render_template("admin_page.html", data=data)

@app.route("/chart/admin/<int:days>", methods=['GET', 'POST'])
@login_required
def admin_show_chart(days):
    algorithms_id = []
    algorithms_name = []
    algorithms_data_id = []
    algorithms_data = []
    counts_data = []
    algorithms = Algorithms.query.all()
    for tup in algorithms:
        algorithms_id.append(tup.id)
        algorithms_name.append(tup.name)
        
    d = datetime.today() - timedelta(days=days)
    data_temp = []
    if days == 0:
        data_temp = db.session.query(Algorithms.id,Algorithms.name,db.func.count(Results.algorithm_id)).outerjoin(Results, Algorithms.id == Results.algorithm_id).group_by(Algorithms.name).order_by(Algorithms.id).all()
    else:
        data_temp = db.session.query(Algorithms.id,Algorithms.name,db.func.count(Results.algorithm_id)).filter(Results.created_at >= d).outerjoin(Results, Algorithms.id == Results.algorithm_id).group_by(Algorithms.name).order_by(Algorithms.id).all()
    for tup in data_temp:
        algorithms_data_id.append(tup[0])
        algorithms_data.append(tup[1])
        counts_data.append(tup[2])
    return jsonify({'algorithms_info': [algorithms_id, algorithms_name], 'algorithms_id': algorithms_data_id, 'algorithms': algorithms_data, 'counts': counts_data})


@app.route("/admin/files", methods=['GET', 'POST'])
@login_required
def admin_file_page():
    check_is_admin()
    files = Files.query.filter_by(is_template=1).order_by(
        desc(Files.created_at)).all()
    form = UploadTemplateForm()
    filename = ''
    if form.file.data:
        filename = secure_filename(
            str(datetime.now())+'-'+form.file.data.filename)
    if form.validate_on_submit():
        data = get_data(form.file.data)
        storage_path = 'uploads\\' + filename
        with open(storage_path, 'w', newline="") as f:
            write = csv.writer(f, delimiter=" ")
            write.writerows(data)
        file = Files(name=filename, storage_path=storage_path,
                     user_id=current_user.id, is_template=1)
        db.session.add(file)
        db.session.commit()
        # form.file.data.save(storage_path)
        flash('Tải lên tập tin mẫu thành công!', category='success')
        return redirect(url_for('admin_file_page'))
    if form.errors != {}:
        # for error in form.errors.values():
        #     flash(error[0], category='danger')
        flash('Lỗi! Xin vui lòng thử lại!', category='danger')
    return render_template("admin_file_page.html", form=form, files=files)


@app.route("/admin/types", methods=['GET', 'POST'])
@login_required
def admin_type_page():
    check_is_admin()
    types = Types.query.all()
    form = TypeForm()
    if form.validate_on_submit():
        type = Types(name=form.name.data)
        db.session.add(type)
        db.session.commit()
        flash('Tạo mới thành công!', category='success')
        return redirect(url_for('admin_type_page'))
    if form.errors != {}:
        # for error in form.errors.values():
        # flash(error[0], category='danger')
        flash('Lỗi! Xin vui lòng thử lại!', category='danger')
    return render_template("admin_type_page.html", form=form, types=types)

@app.route("/admin/types/<int:id>", methods=['GET', 'POST'])
@login_required
def admin_type_page_edit(id):
    check_is_admin()
    form = TypeForm()
    type = Types.query.filter_by(id=id).first()
    if form.validate_on_submit():
        type.name = form.name.data
        db.session.commit()
        flash(f'Chỉnh sửa thành công! ', category='success')
        return redirect(url_for('admin_type_page'))
    if form.errors != {}:
        # for error in form.errors.values():
        #     flash(error[0], category='danger')
        flash('Lỗi! Xin vui lòng thử lại!', category='danger')
    return render_template("admin_type_page_edit.html", form=form, type = type)

@app.route("/admin/algorithms", methods=['GET', 'POST'])
@login_required
def admin_algorithm_page():
    check_is_admin()
    algorithms = Algorithms.query.all()
    form = AlgorithmForm()
    if form.validate_on_submit():
        algorithm = Algorithms(name=form.name.data,
                               description=form.description.data, link=form.link.data)
        db.session.add(algorithm)
        db.session.commit()
        flash(f'Tạo mới thành công! ', category='success')
        return redirect(url_for('admin_algorithm_page'))
    if form.errors != {}:
        # for error in form.errors.values():
        #     flash(error[0], category='danger')
        flash('Lỗi! Xin vui lòng thử lại!', category='danger')
    return render_template("admin_algorithm_page.html", algorithms=algorithms, form=form)


@app.route("/admin/algorithms/<int:id>", methods=['GET', 'POST'])
@login_required
def admin_algorithm_page_edit(id):
    check_is_admin()
    form = AlgorithmForm()
    algorithm = Algorithms.query.filter_by(id=id).first()
    if form.validate_on_submit():
        algorithm.name = form.name.data
        algorithm.description = form.description.data
        algorithm.link = form.link.data
        db.session.commit()
        flash(f'Chỉnh sửa thành công! ', category='success')
        return redirect(url_for('admin_algorithm_page'))
    if form.errors != {}:
        # for error in form.errors.values():
        #     flash(error[0], category='danger')
        flash('Lỗi! Xin vui lòng thử lại!', category='danger')
    form.description.data = algorithm.description
    return render_template("admin_algorithm_page_edit.html", form=form, algorithm = algorithm)

@app.route("/admin/files/<int:id>", methods=['GET', 'POST'])
@login_required
def admin_file_page_show(id):
    check_is_admin()
    file = Files.query.filter_by(id=id).first()
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    path = os.path.join(parent_path, file.storage_path)
    data = get_data(path)
    return render_template("show_sample.html", data=data)


# def show(id):
#     file = Files.query.filter_by(id=id).first()
#     parent_path = pathlib.Path(__file__).parent.parent.resolve()
#     path = os.path.join(parent_path, file.link)
#     data = []
#     max = 1
#     with open(path) as f:
#         d = csv.reader(f)
#         for row in d:
#             row = row[0].split(" ")
#             if max < len(row): max = len(row)
#             data.append(list(row[1:]))
#     return render_template("show.html", data=data, max=max)

# algorithms_admin
# @app.route("/algorithm", methods=['GET', 'POST'])
# @login_required
# def algorithm_page():
#     check_is_admin()
#     algorithms = Algorithms.query.all()
#     form = AlgorithmForm()
#     if form.validate_on_submit():
#         algorithm = Algorithms(name=form.name.data,
#                                description=form.description.data)
#         db.session.add(algorithm)
#         db.session.commit()
#         flash(f'Tạo mới thành công! ', category='success')
#         return redirect(url_for('algorithm_page'))
#     if form.errors != {}:
#         for error in form.errors.values():
#             flash(error[0], category='danger')
#     return render_template("algorithm_page.html", algorithms=algorithms, form=form)


# @app.route("/algorithm/<int:id>", methods=['GET', 'POST'])
# @login_required
# def algorithm_edit_page(id):
#     check_is_admin()
#     form = AlgorithmForm()
#     algorithm = Algorithms.query.filter_by(id=id).first()
#     if form.validate_on_submit():
#         algorithm.name = form.name.data
#         algorithm.description = form.description.data
#         print(form.description.data)
#         db.session.commit()
#         flash(f'Chỉnh sửa thành công! ', category='success')
#         return redirect(url_for('algorithm_page'))
#     if form.errors != {}:
#         for error in form.errors.values():
#             flash(error[0], category='danger')
#     form.description.data = algorithm.description
#     return render_template("algorithm_edit_page.html", form=form, algorithm = algorithm)
