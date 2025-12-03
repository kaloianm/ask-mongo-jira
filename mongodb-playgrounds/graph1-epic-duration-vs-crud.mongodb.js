// MongoDB aggregation pipeline for Graph 1: Epic Duration vs CRUD Percentage
//
// This pipeline joins jira_epics with code_analysis to calculate:
// - Epic duration in weeks (end_date - start_date)
// - CRUD percentage = Catalog CRUD / (Catalog CRUD + Catalog DDL + Catalog Implementation Change)

use('ask-mongo-jira');

db.jira_epics.aggregate([
    // Join jira_epics with code_analysis
    {
        $lookup: {
            from: 'code_analysis',
            localField: 'key',
            foreignField: 'epic_key',
            as: 'analyses'
        }
    },
    // Filter out epics with no analysis data
    {
        $match: {
            'analyses': { $ne: [] },
            'start_date': { $exists: true },
            'end_date': { $exists: true }
        }
    },
    // Project initial fields
    {
        $project: {
            epic_key: '$key',
            name: '$name',
            start_date: '$start_date',
            end_date: '$end_date',
            analyses: '$analyses'
        }
    },
    // Unwind analyses to process each classification
    {
        $unwind: '$analyses'
    },
    // Group by epic and count classifications
    {
        $group: {
            _id: {
                epic_key: '$epic_key',
                name: '$name',
                start_date: '$start_date',
                end_date: '$end_date'
            },
            catalog_crud: {
                $sum: {
                    $cond: [
                        { $eq: ['$analyses.classification', 'Catalog CRUD'] },
                        1, 0
                    ]
                }
            },
            catalog_ddl: {
                $sum: {
                    $cond: [
                        { $eq: ['$analyses.classification', 'Catalog DDL'] },
                        1, 0
                    ]
                }
            },
            catalog_implementation_change: {
                $sum: {
                    $cond: [
                        { $eq: ['$analyses.classification', 'Catalog Implementation Change'] },
                        1, 0
                    ]
                }
            },
            total_classifications: { $sum: 1 }
        }
    },
    // Calculate duration in weeks and CRUD percentage
    {
        $project: {
            epic_key: '$_id.epic_key',
            name: '$_id.name',
            start_date: '$_id.start_date',
            end_date: '$_id.end_date',
            catalog_crud: '$catalog_crud',
            catalog_ddl: '$catalog_ddl',
            catalog_implementation_change: '$catalog_implementation_change',
            total_classifications: '$total_classifications',
            catalog_total: {
                $add: ['$catalog_crud', '$catalog_ddl', '$catalog_implementation_change']
            },
            duration_days: {
                $divide: [
                    { $subtract: ['$_id.end_date', '$_id.start_date'] },
                    86400000  // Convert milliseconds to days
                ]
            }
        }
    },
    {
        $project: {
            epic_key: 1,
            name: 1,
            start_date: 1,
            end_date: 1,
            catalog_crud: 1,
            catalog_ddl: 1,
            catalog_implementation_change: 1,
            total_classifications: 1,
            catalog_total: 1,
            duration_weeks: { $divide: ['$duration_days', 7] },
            crud_percentage: {
                $cond: [
                    { $gt: ['$catalog_total', 0] },
                    { $multiply: [{ $divide: ['$catalog_crud', '$catalog_total'] }, 100] },
                    0
                ]
            }
        }
    },
    // Filter out epics with no catalog operations
    {
        $match: {
            catalog_total: { $gt: 0 }
        }
    },
    {
        $sort: { duration_weeks: 1 }
    }
]);