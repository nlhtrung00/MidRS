$(document).ready(function () {
	let last = $("input[name='type']:last").val();
	$("input[name='type']").click(function () {
		let index = $("input[name='type']:checked").val();
		if (index == last) {
			$('#other').text('');
			$('#other').css('display', 'block');
			$('#other').prop('required', true);
		} else {
			$('#other').css('display', 'none');
			$('#other').prop('required', false);
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
