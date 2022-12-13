$(document).ready(function () {
	$('#toggleForm').click(function () {
		if (!$('form').is(':visible')) {
			$(this).text('Ẩn biểu mẫu');
			$('form').show();
		} else {
			$(this).text('Hiển thị biểu mẫu');
			$('form').hide();
		}
	});
	$("input[name='select']").click(function () {
		let index = $("input[name='select']:checked").val();
		if (
			index == 1 ||
			index == 3 ||
			index == 4 ||
			index == 5 ||
			index == 6 ||
			index == 7 ||
			index == 8
		) {
			$('#knn').val('');
			$('.knn').css('display', 'block');
			$('#knn').prop('required', true);
		} else {
			$('.knn').css('display', 'none');
			$('#knn').prop('required', false);
		}
	});
});
