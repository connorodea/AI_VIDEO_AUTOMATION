// ui/web/static/js/main.js

$(document).ready(function() {
    // Variables for project operations
    let deleteProjectId = null;
    let resumeProjectId = null;

    // Project status polling
    if ($('.project-status-page').length > 0) {
        const projectId = $('.project-status-page').data('project-id');
        pollProjectStatus(projectId);
    }

    // Handle delete project button
    $('.delete-project').click(function() {
        deleteProjectId = $(this).data('project-id');
        $('#deleteProjectModal').modal('show');
    });

    // Handle confirm delete
    $('#confirmDelete').click(function() {
        if (deleteProjectId) {
            deleteProject(deleteProjectId);
        }
    });

    // Handle resume project button
    $('.resume-project').click(function() {
        resumeProjectId = $(this).data('project-id');
        $('#resumeProjectModal').modal('show');
    });

    // Handle confirm resume
    $('#confirmResume').click(function() {
        if (resumeProjectId) {
            resumeProject(resumeProjectId);
        }
    });

    // Handle advanced options in stage selection
    $('#advancedOptions input[name="stages"]').change(function() {
        updateStageSequence();
    });

    // Update progress bars with animation
    $('.progress-bar').each(function() {
        const width = $(this).attr('aria-valuenow') + '%';
        $(this).css('width', width);
    });
});

/**
 * Poll the project status API
 */
function pollProjectStatus(projectId) {
    const pollInterval = 3000; // 3 seconds

    function updateStatus() {
        $.ajax({
            url: `/api/project/${projectId}/status`,
            method: 'GET',
            dataType: 'json',
            success: function(data) {
                // Update status
                $('#project-status').text(data.status.toUpperCase());
                
                // Update progress bar
                const progress = data.progress;
                $('#progress-bar')
                    .css('width', progress + '%')
                    .attr('aria-valuenow', progress)
                    .text(progress + '%');
                
                // Update stage statuses
                if (data.stages) {
                    for (const stage in data.stages) {
                        const stageInfo = data.stages[stage];
                        const statusElement = $(`#stage-${stage}`);
                        
                        if (statusElement.length) {
                            // Update status text and class
                            let badgeClass = 'secondary';
                            if (stageInfo.status === 'completed') badgeClass = 'success';
                            else if (stageInfo.status === 'failed') badgeClass = 'danger';
                            else if (stageInfo.status === 'running') badgeClass = 'primary';
                            else if (stageInfo.status === 'skipped') badgeClass = 'warning';
                            
                            statusElement
                                .text(stageInfo.status.toUpperCase())
                                .removeClass('bg-secondary bg-success bg-danger bg-primary bg-warning')
                                .addClass(`bg-${badgeClass}`);
                        }
                    }
                }
                
                // Check if project is completed or failed
                if (data.status === 'completed' || data.status === 'failed' || data.status === 'partially_completed') {
                    // Show final output section if completed
                    if (data.status === 'completed' || data.status === 'partially_completed') {
                        $('#output-section').removeClass('d-none');
                        
                        // Try to update output links if available
                        if (data.stages && data.stages.final_export && data.stages.final_export.output) {
                            const output = data.stages.final_export.output;
                            if (output.output_path) {
                                $('#final-video-link').attr('href', getMediaUrl(output.output_path)).removeClass('disabled');
                            }
                        } else if (data.stages && data.stages.video_assembly && data.stages.video_assembly.output) {
                            const output = data.stages.video_assembly.output;
                            if (output.output_path) {
                                $('#draft-video-link').attr('href', getMediaUrl(output.output_path)).removeClass('disabled');
                            }
                        }
                    }
                    
                    // Show error message if failed
                    if (data.status === 'failed') {
                        $('#error-section').removeClass('d-none');
                        if (data.error) {
                            $('#error-message').text(data.error);
                        }
                    }
                    
                    // Stop polling
                    return;
                }
                
                // Continue polling if not completed or failed
                setTimeout(updateStatus, pollInterval);
            },
            error: function(xhr, status, error) {
                console.error('Error polling project status:', error);
                // Retry after a delay
                setTimeout(updateStatus, pollInterval * 2);
            }
        });
    }

    // Start polling
    updateStatus();
}

/**
 * Delete a project
 */
function deleteProject(projectId) {
    $.ajax({
        url: `/api/project/${projectId}/delete`,
        method: 'POST',
        success: function(data) {
            if (data.success) {
                // Close modal and refresh page
                $('#deleteProjectModal').modal('hide');
                window.location.reload();
            } else {
                alert('Failed to delete project. Please try again.');
            }
        },
        error: function(xhr, status, error) {
            alert('Error deleting project: ' + error);
            $('#deleteProjectModal').modal('hide');
        }
    });
}

/**
 * Resume a project
 */
function resumeProject(projectId) {
    const formData = $('#resumeForm').serialize();
    
    $.ajax({
        url: `/api/project/${projectId}/resume`,
        method: 'POST',
        data: formData,
        success: function(data) {
            if (data.success) {
                // Close modal and redirect to tracking page
                $('#resumeProjectModal').modal('hide');
                window.location.href = `/project/${data.tracking_id}`;
            } else {
                alert('Failed to resume project. Please try again.');
            }
        },
        error: function(xhr, status, error) {
            alert('Error resuming project: ' + error);
            $('#resumeProjectModal').modal('hide');
        }
    });
}

/**
 * Update the stage sequence based on selected checkboxes
 */
function updateStageSequence() {
    const stages = [];
    $('#advancedOptions input[name="stages"]:checked').each(function() {
        stages.push($(this).val());
    });

    // Add the sequence to a hidden input
    if ($('#stage_sequence').length === 0) {
        $('<input>').attr({
            type: 'hidden',
            id: 'stage_sequence',
            name: 'stage_sequence'
        }).appendTo('form');
    }
    
    $('#stage_sequence').val(stages.join(','));
}

/**
 * Get media URL from path
 */
function getMediaUrl(path) {
    // Extract the relative path from the full path
    const parts = path.split('/');
    const mediaPath = parts.slice(parts.indexOf('output')).join('/');
    return `/media/${mediaPath}`;
}
